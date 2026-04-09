import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from tools.inference_service import (
        MODEL_CACHE,
        PROMPT_CACHE,
        audio_to_base64_wav,
        audio_to_wav_bytes,
        clear_memory,
        format_status_text,
        get_runtime_config,
        synthesize,
    )
except ImportError:
    from inference_service import (
        MODEL_CACHE,
        PROMPT_CACHE,
        audio_to_base64_wav,
        audio_to_wav_bytes,
        clear_memory,
        format_status_text,
        get_runtime_config,
        synthesize,
    )


app = FastAPI(title="GLMTTS API", version="1.0.0")

API_FILES_ROOT = Path(
    os.environ.get(
        "GLMTTS_API_FILES_ROOT",
        Path(__file__).resolve().parent.parent / "api_files",
    )
).resolve()
VOICE_FILES_DIR = API_FILES_ROOT / "voices"
GENERATED_FILES_DIR = API_FILES_ROOT / "generated"
PUBLIC_BASE_URL = os.environ.get("GLMTTS_PUBLIC_BASE_URL", "").rstrip("/")
MONGO_URI = os.environ.get("GLMTTS_MONGO_URI", "mongodb://127.0.0.1:27017")
MONGO_DB_NAME = os.environ.get("GLMTTS_MONGO_DB", "glmtts")
MONGO_VOICE_COLLECTION = os.environ.get("GLMTTS_MONGO_VOICE_COLLECTION", "voices")
_MONGO_CLIENT = None
_MONGO_COLLECTION = None
VOICE_INDEX_FILE = VOICE_FILES_DIR / "voice_index.json"

VOICE_FILES_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_FILES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/files", StaticFiles(directory=str(API_FILES_ROOT)), name="media_files")


class GenerateTTSRequest(BaseModel):
    voice_id: str
    input_text: str
    seed: int = 42
    sample_rate: int = 24000
    use_cache: bool = True


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _read_voice_index():
    if not VOICE_INDEX_FILE.exists():
        return {}
    with open(VOICE_INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_voice_index(data):
    VOICE_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VOICE_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _try_get_voice_collection():
    global _MONGO_CLIENT, _MONGO_COLLECTION
    if _MONGO_COLLECTION is not None:
        return _MONGO_COLLECTION

    try:
        from pymongo import MongoClient
    except ImportError:
        return None

    try:
        _MONGO_CLIENT = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1000)
        _MONGO_CLIENT.admin.command("ping")
        _MONGO_COLLECTION = _MONGO_CLIENT[MONGO_DB_NAME][MONGO_VOICE_COLLECTION]
        return _MONGO_COLLECTION
    except Exception:
        _MONGO_CLIENT = None
        _MONGO_COLLECTION = None
        return None


def _build_public_url(request: Request, relative_path: str):
    clean_path = relative_path.lstrip("/")
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}/files/{clean_path}"
    return str(request.base_url).rstrip("/") + f"/files/{clean_path}"


def _voice_file_relative_path(voice_id: str, filename: str):
    return f"voices/{voice_id}/{filename}"


def _generated_file_relative_path(filename: str):
    return f"generated/{filename}"


def _write_binary_file(file_path: Path, data: bytes):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(data)


async def _materialize_clone_audio(reference_audio, reference_audio_path, voice_id):
    if reference_audio is None and not reference_audio_path:
        raise HTTPException(
            status_code=400,
            detail="Either reference_audio or reference_audio_path is required.",
        )

    if reference_audio is not None:
        suffix = Path(reference_audio.filename or "reference.wav").suffix or ".wav"
        relative_path = _voice_file_relative_path(voice_id, f"reference{suffix}")
        target_path = API_FILES_ROOT / relative_path
        _write_binary_file(target_path, await reference_audio.read())
        return target_path, relative_path

    source_path = Path(reference_audio_path).expanduser().resolve()
    if not source_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"reference_audio_path does not exist: {reference_audio_path}",
        )
    suffix = source_path.suffix or ".wav"
    relative_path = _voice_file_relative_path(voice_id, f"reference{suffix}")
    target_path = API_FILES_ROOT / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)
    return target_path, relative_path


def _get_voice_document(voice_id: str):
    collection = _try_get_voice_collection()
    if collection is not None:
        doc = collection.find_one({"_id": voice_id})
        if doc is None:
            raise HTTPException(status_code=404, detail=f"voice_id not found: {voice_id}")
        return doc

    voice_index = _read_voice_index()
    doc = voice_index.get(voice_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"voice_id not found: {voice_id}")
    return doc


def _save_voice_document(doc):
    collection = _try_get_voice_collection()
    if collection is not None:
        collection.insert_one(doc)
        return "mongo"

    voice_index = _read_voice_index()
    voice_index[doc["_id"]] = doc
    _write_voice_index(voice_index)
    return "json"


def _save_generated_wav(request: Request, sample_rate: int, audio_int16):
    generated_id = uuid4().hex
    relative_path = _generated_file_relative_path(f"{generated_id}.wav")
    output_path = API_FILES_ROOT / relative_path
    _write_binary_file(output_path, audio_to_wav_bytes(sample_rate, audio_int16))
    return generated_id, relative_path, _build_public_url(request, relative_path)


def _success_payload(data, message="success"):
    return {"code": 0, "message": message, "data": data}


async def _materialize_prompt_audio(prompt_audio, prompt_audio_path):
    if prompt_audio is None and not prompt_audio_path:
        raise HTTPException(status_code=400, detail="Either prompt_audio or prompt_audio_path is required.")

    if prompt_audio is None:
        return prompt_audio_path, None

    suffix = Path(prompt_audio.filename or "prompt.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(await prompt_audio.read())
        return tmp_file.name, tmp_file.name


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": MODEL_CACHE["loaded"],
        "cached_sample_rate": MODEL_CACHE["sample_rate"],
        "prompt_cache_size": len(PROMPT_CACHE),
        "config": get_runtime_config(),
    }


@app.post("/clear-memory")
def clear_memory_endpoint():
    return {
        "status": "ok",
        "message": clear_memory(),
    }


@app.post("/voices/clone")
async def clone_voice_endpoint(
    request: Request,
    prompt_text: str = Form(...),
    reference_audio_path: str | None = Form(None),
    voice_name: str = Form(""),
    reference_audio: UploadFile | None = File(None),
):
    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required.")

    voice_id = uuid4().hex
    saved_audio_path, relative_audio_path = await _materialize_clone_audio(
        reference_audio=reference_audio,
        reference_audio_path=reference_audio_path,
        voice_id=voice_id,
    )

    now_iso = _utc_now_iso()
    doc = {
        "_id": voice_id,
        "voice_id": voice_id,
        "voice_name": voice_name.strip(),
        "prompt_text": prompt_text,
        "reference_audio_path": str(saved_audio_path),
        "reference_audio_relative_path": relative_audio_path,
        "reference_audio_url": _build_public_url(request, relative_audio_path),
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    storage_backend = _save_voice_document(doc)

    return JSONResponse(
        content=_success_payload(
            {
                "voice_id": voice_id,
                "voice_name": doc["voice_name"],
                "prompt_text": prompt_text,
                "reference_audio_url": doc["reference_audio_url"],
                "storage_backend": storage_backend,
            }
        )
    )


@app.post("/tts/generate")
async def generate_tts_endpoint(
    payload: GenerateTTSRequest,
):
    voice_id = payload.voice_id.strip()
    input_text = payload.input_text.strip()
    if not voice_id:
        raise HTTPException(status_code=400, detail="voice_id is required.")
    if not input_text:
        raise HTTPException(status_code=400, detail="input_text is required.")

    voice_doc = _get_voice_document(voice_id)

    try:
        result = synthesize(
            prompt_text=voice_doc["prompt_text"],
            prompt_audio_path=voice_doc["reference_audio_path"],
            input_text=input_text,
            seed=payload.seed,
            sample_rate=payload.sample_rate,
            use_cache=payload.use_cache,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    status_text = format_status_text(result)
    wav_bytes = audio_to_wav_bytes(result["sample_rate"], result["audio_int16"])
    headers = {
        "X-Voice-Id": voice_id,
        "X-Elapsed-Seconds": f"{result['elapsed_seconds']:.6f}",
        "X-Audio-Seconds": f"{result['audio_seconds']:.6f}",
        "X-RTF": f"{result['rtf']:.6f}",
        "X-Status-Text": status_text,
    }

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers=headers,
    )


@app.post("/synthesize")
async def synthesize_endpoint(
    prompt_text: str = Form(""),
    input_text: str = Form(...),
    seed: int = Form(42),
    sample_rate: int = Form(24000),
    use_cache: str = Form("true"),
    prompt_audio_path: str | None = Form(None),
    response_format: str = Form("json"),
    prompt_audio: UploadFile | None = File(None),
):
    temp_path = None
    prompt_source = None
    try:
        prompt_source, temp_path = await _materialize_prompt_audio(
            prompt_audio=prompt_audio,
            prompt_audio_path=prompt_audio_path,
        )
        result = synthesize(
            prompt_text=prompt_text,
            prompt_audio_path=prompt_source,
            input_text=input_text,
            seed=seed,
            sample_rate=sample_rate,
            use_cache=_parse_bool(use_cache, default=True),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    wav_bytes = audio_to_wav_bytes(result["sample_rate"], result["audio_int16"])
    status_text = format_status_text(result)
    headers = {
        "X-Elapsed-Seconds": f"{result['elapsed_seconds']:.6f}",
        "X-Audio-Seconds": f"{result['audio_seconds']:.6f}",
        "X-RTF": f"{result['rtf']:.6f}",
        "X-Status-Text": status_text,
    }

    if response_format.strip().lower() == "wav":
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

    payload = {
        "status": "ok",
        "sample_rate": result["sample_rate"],
        "audio_samples": result["audio_samples"],
        "audio_seconds": result["audio_seconds"],
        "elapsed_seconds": result["elapsed_seconds"],
        "rtf": result["rtf"],
        "num_output_tokens": result["num_output_tokens"],
        "normalized_prompt_text": result["normalized_prompt_text"],
        "normalized_input_text": result["normalized_input_text"],
        "prompt_cache_size": result["prompt_cache_size"],
        "status_text": status_text,
        "config": result["config"],
        "audio_base64_wav": audio_to_base64_wav(
            result["sample_rate"], result["audio_int16"]
        ),
    }
    return JSONResponse(content=payload, headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("GLMTTS_API_PORT", "8048")),
    )
