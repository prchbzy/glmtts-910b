import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

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


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
