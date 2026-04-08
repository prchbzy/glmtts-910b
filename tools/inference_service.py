import base64
import gc
import io
import logging
import os
import time
import wave

import numpy as np
import torch

from glmtts_inference import (
    DEVICE,
    PROMPT_CACHE,
    generate_long,
    get_prompt_cache_entry,
    load_models,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


SERVICE_LLM_BACKEND = os.environ.get("GLMTTS_GRADIO_LLM_BACKEND", "vllm")
SERVICE_HF_ATTN_IMPLEMENTATION = os.environ.get(
    "GLMTTS_GRADIO_HF_ATTN_IMPLEMENTATION", None
)
SERVICE_HF_GRAPH_DECODE = _env_flag("GLMTTS_GRADIO_HF_GRAPH_DECODE", False)
SERVICE_HF_GRAPH_MAX_CACHE_LEN = int(
    os.environ.get("GLMTTS_GRADIO_HF_GRAPH_MAX_CACHE_LEN", "1536")
)
SERVICE_HF_GRAPH_WARMUP_ITERS = int(
    os.environ.get("GLMTTS_GRADIO_HF_GRAPH_WARMUP_ITERS", "2")
)
SERVICE_SAMPLE_METHOD = os.environ.get("GLMTTS_GRADIO_SAMPLE_METHOD", "ras")
SERVICE_SAMPLING_TOP_P = float(os.environ.get("GLMTTS_GRADIO_SAMPLE_TOP_P", "0.8"))
SERVICE_SAMPLING_TOP_K = int(os.environ.get("GLMTTS_GRADIO_SAMPLE_TOP_K", "25"))
SERVICE_SAMPLING_TEMPERATURE = float(
    os.environ.get("GLMTTS_GRADIO_SAMPLE_TEMPERATURE", "0.7")
)


MODEL_CACHE = {
    "loaded": False,
    "sample_rate": None,
    "components": None,
}


def get_runtime_config():
    return {
        "hf_attn_implementation": SERVICE_HF_ATTN_IMPLEMENTATION,
        "hf_graph_decode": SERVICE_HF_GRAPH_DECODE,
        "hf_graph_max_cache_len": SERVICE_HF_GRAPH_MAX_CACHE_LEN,
        "hf_graph_warmup_iters": SERVICE_HF_GRAPH_WARMUP_ITERS,
        "sample_method": SERVICE_SAMPLE_METHOD,
        "top_p": SERVICE_SAMPLING_TOP_P,
        "top_k": SERVICE_SAMPLING_TOP_K,
        "temperature": SERVICE_SAMPLING_TEMPERATURE,
    }


def get_models(use_phoneme=False, sample_rate=24000):
    if MODEL_CACHE["loaded"] and MODEL_CACHE["sample_rate"] == sample_rate:
        return MODEL_CACHE["components"]

    logging.info("Loading models with sample_rate=%s...", sample_rate)

    if MODEL_CACHE["components"]:
        del MODEL_CACHE["components"]
        gc.collect()
        torch.cuda.empty_cache()

    frontend, text_frontend, speech_tokenizer, llm, flow = load_models(
        use_phoneme=use_phoneme,
        sample_rate=sample_rate,
    )

    MODEL_CACHE["components"] = (frontend, text_frontend, speech_tokenizer, llm, flow)
    MODEL_CACHE["sample_rate"] = sample_rate
    MODEL_CACHE["loaded"] = True
    logging.info("Models loaded successfully.")
    return MODEL_CACHE["components"]


def _audio_to_int16(tts_speech):
    audio_data = tts_speech.squeeze().cpu().numpy()
    audio_data = np.clip(audio_data, -1.0, 1.0)
    return (audio_data * 32767.0).astype(np.int16)


def audio_to_wav_bytes(sample_rate, audio_int16):
    audio_int16 = np.asarray(audio_int16, dtype=np.int16)
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        return buffer.getvalue()


def audio_to_base64_wav(sample_rate, audio_int16):
    return base64.b64encode(audio_to_wav_bytes(sample_rate, audio_int16)).decode("ascii")


def format_status_text(result):
    return (
        f"graph_decode={result['config']['hf_graph_decode']} "
        f"attn={result['config']['hf_attn_implementation']} | "
        f"elapsed={result['elapsed_seconds']:.3f}s "
        f"audio={result['audio_seconds']:.3f}s "
        f"rtf={result['rtf']:.4f}"
    )


def synthesize(prompt_text, prompt_audio_path, input_text, seed, sample_rate, use_cache=True):
    if not input_text:
        raise ValueError("Please provide text to synthesize.")
    if not prompt_audio_path:
        raise ValueError("Please upload a prompt audio file.")

    start_time = time.perf_counter()
    frontend, text_frontend, _, llm, flow = get_models(
        use_phoneme=True, sample_rate=sample_rate
    )

    normalized_input_text = text_frontend.text_normalize(input_text)
    prompt_entry = get_prompt_cache_entry(
        frontend=frontend,
        text_frontend=text_frontend,
        prompt_text_raw=prompt_text,
        prompt_speech_path=prompt_audio_path,
        sample_rate=sample_rate,
    )
    normalized_prompt_text = prompt_entry["prompt_text"] + " "

    logging.info("Normalized Prompt: %s", normalized_prompt_text)
    logging.info("Normalized Input: %s", normalized_input_text)
    logging.info(
        "Shared inference config: backend=%s hf_attn_implementation=%s hf_graph_decode=%s hf_graph_max_cache_len=%s hf_graph_warmup_iters=%s sample_method=%s top_p=%s top_k=%s temperature=%s",
        SERVICE_LLM_BACKEND,
        SERVICE_HF_ATTN_IMPLEMENTATION,
        SERVICE_HF_GRAPH_DECODE,
        SERVICE_HF_GRAPH_MAX_CACHE_LEN,
        SERVICE_HF_GRAPH_WARMUP_ITERS,
        SERVICE_SAMPLE_METHOD,
        SERVICE_SAMPLING_TOP_P,
        SERVICE_SAMPLING_TOP_K,
        SERVICE_SAMPLING_TEMPERATURE,
    )

    cache_speech_token_list = [
        tokens.copy() for tokens in prompt_entry["cache_speech_token"]
    ]
    cache = {
        "cache_text": [normalized_prompt_text],
        "cache_text_token": [prompt_entry["prompt_text_token"]],
        "cache_speech_token": cache_speech_token_list,
        "use_cache": use_cache,
    }

    tts_speech, _, output_token_list, text_tn_dict = generate_long(
        frontend=frontend,
        text_frontend=text_frontend,
        llm=llm,
        flow=flow,
        text_info=["", normalized_input_text],
        cache=cache,
        embedding=prompt_entry["embedding"],
        flow_prompt_token=prompt_entry["flow_prompt_token"],
        speech_feat=prompt_entry["speech_feat"],
        sample_method=SERVICE_SAMPLE_METHOD,
        sampling_top_p=SERVICE_SAMPLING_TOP_P,
        sampling_top_k=SERVICE_SAMPLING_TOP_K,
        sampling_temperature=SERVICE_SAMPLING_TEMPERATURE,
        seed=seed,
        device=DEVICE,
        use_phoneme=False,
    )

    elapsed_seconds = time.perf_counter() - start_time
    audio_int16 = _audio_to_int16(tts_speech)
    audio_samples = int(audio_int16.shape[0]) if audio_int16.ndim > 0 else 0
    audio_seconds = audio_samples / float(sample_rate) if sample_rate > 0 else 0.0
    rtf = elapsed_seconds / audio_seconds if audio_seconds > 0 else float("inf")

    return {
        "sample_rate": sample_rate,
        "audio_int16": audio_int16,
        "audio_samples": audio_samples,
        "audio_seconds": audio_seconds,
        "elapsed_seconds": elapsed_seconds,
        "rtf": rtf,
        "normalized_prompt_text": normalized_prompt_text,
        "normalized_input_text": normalized_input_text,
        "num_output_tokens": len(output_token_list),
        "text_tn_dict": text_tn_dict,
        "prompt_cache_size": len(PROMPT_CACHE),
        "config": get_runtime_config(),
    }


def run_gradio_inference(prompt_text, prompt_audio_path, input_text, seed, sample_rate, use_cache=True):
    result = synthesize(
        prompt_text=prompt_text,
        prompt_audio_path=prompt_audio_path,
        input_text=input_text,
        seed=seed,
        sample_rate=sample_rate,
        use_cache=use_cache,
    )
    return (result["sample_rate"], result["audio_int16"]), format_status_text(result)


def clear_memory():
    if MODEL_CACHE["components"]:
        del MODEL_CACHE["components"]
    MODEL_CACHE["components"] = None
    MODEL_CACHE["loaded"] = False
    MODEL_CACHE["sample_rate"] = None
    PROMPT_CACHE.clear()

    gc.collect()
    torch.cuda.empty_cache()
    return "Memory cleared. Models will reload on next inference."
