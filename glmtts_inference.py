# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
# Authors: Jiayan Cui, Zhihan Yang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import multiprocessing
import os
import sys

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_V1"] = "0"
os.environ["VLLM_USE_MODELSCOPE"] = "false"
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"

import torch

try:
    torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 0)
    import torch.utils._triton

    if hasattr(torch.utils._triton, "_device_supports_tma"):
        torch.utils._triton._device_supports_tma = (
            lambda *args, **kwargs: False
        )
    if hasattr(torch.utils._triton, "has_triton_experimental_host_tma"):
        torch.utils._triton.has_triton_experimental_host_tma = (
            lambda *args, **kwargs: False
        )
except Exception as e:
    print(f"import failed: {e}")

import torchaudio
import tqdm
import logging
import time
import argparse
import json
import math
from pathlib import Path
from functools import partial

from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
from utils import file_utils, seed_util
from utils import tts_model_util, yaml_util
from transformers import AutoTokenizer
from llm.glmtts import GLMTTS
from utils.audio import mel_spectrogram
from vllm import LLM, SamplingParams
from vllm.sampling_params import SamplingParams as VllmSamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LLM_SEQ_INP_LEN = 750
TOKEN_RATE = 25
EOS_TOKEN_ID_AFTER_MINUS_BOS = None
PROMPT_CACHE = {}

# Configure logging
# Conservative RAS tuning for the restored old vLLM path.
# Goal: reduce breath/noise artifacts without touching chunk/cache behavior.
RAS_TOP_P = 0.8
RAS_TOP_K = 25
RAS_WIN_SIZE = 10
RAS_TAU_R = 0.20
RAS_TEMPERATURE = 1.0

# HF stepwise experiment params to compare against the vLLM path.
HF_SAMPLE_TOP_P = 0.8
HF_SAMPLE_TOP_K = 25
HF_SAMPLE_TEMPERATURE = 0.7

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)


def _elapsed_seconds(start_time):
    return time.perf_counter() - start_time


def _write_token_dump(
    dump_dir,
    uttid,
    backend,
    sample_method,
    chunk_records,
):
    if not dump_dir:
        return

    safe_uttid = uttid or "interactive"
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    file_path = dump_path / f"{safe_uttid}.{backend}.{sample_method}.tokens.json"
    payload = {
        "uttid": safe_uttid,
        "backend": backend,
        "sample_method": sample_method,
        "num_chunks": len(chunk_records),
        "chunks": chunk_records,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info("[token_dump] wrote %s", file_path)


def set_npu_jit(enabled: bool):
    if NPU_AVAILABLE and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_compile_mode(jit_compile=enabled)


if NPU_AVAILABLE:
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

    # 备份原函数
    _orig_torch_fbank = WhisperFeatureExtractor._torch_extract_fbank_features
    _orig_hann_window = torch.hann_window

    def _torch_fbank_cpu_only(self, waveform, device="cpu"):
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().to("cpu", dtype=torch.float32)
        return _orig_torch_fbank(self, waveform, device="cpu")

    def _hann_window_force_fp32(window_length, *args, **kwargs):
        kwargs["dtype"] = torch.float32
        return _orig_hann_window(window_length, *args, **kwargs)

    torch.hann_window = _hann_window_force_fp32
    WhisperFeatureExtractor._torch_extract_fbank_features = _torch_fbank_cpu_only


def get_special_token_ids(tokenize_fn):
    """
    Get special token IDs based on the tokenizer name.
    """
    _special_token_ids = {
        "ats": "<|audio_0|>",
        "ate": "<|audio_32767|>",
        "boa": "<|begin_of_audio|>",
        "eoa": "<|user|>",
        "pad": "<|endoftext|>",
    }

    special_token_ids = {}

    # Validation
    endoftext_id = tokenize_fn("<|endoftext|>")[0]
    for k, v in _special_token_ids.items():
        __ids = tokenize_fn(v)
        # Check 1: Special token length must be 1
        if len(__ids) != 1:
            raise AssertionError(
                f"Token '{k}' ({v}) encoded to multiple tokens: {__ids}"
            )
        # Check 2: Special token ID must be >= endoftext_id
        if __ids[0] < endoftext_id:
            raise AssertionError(
                f"Token '{k}' ({v}) ID {__ids[0]} is smaller than endoftext ID {endoftext_id}"
            )

        special_token_ids[k] = __ids[0]

    return special_token_ids

def _assert_shape_and_get_len(token):
    assert token.ndim == 2 and token.shape[0] == 1
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(token.device)
    return token_len

def load_frontends(speech_tokenizer, sample_rate=24000, use_phoneme=False, frontend_dir="frontend"):
    if sample_rate == 32000:
        feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=640, n_fft=2560, num_mels=80, win_size=2560, fmin=0, fmax=8000, center=False)
        logging.info("Configured for 32kHz frontend.")
    elif sample_rate == 24000:
        feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=480, n_fft=1920, num_mels=80, win_size=1920, fmin=0, fmax=8000, center=False)
        logging.info("Configured for 24kHz frontend.")
    else:
        raise ValueError(f"Unsupported sampling_rate: {sample_rate}")

    glm_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join('ckpt', 'vq32k-phoneme-tokenizer'), trust_remote_code=True
    )

    tokenize_fn = lambda text: glm_tokenizer.encode(text)

    frontend = TTSFrontEnd(
        tokenize_fn,
        speech_tokenizer,
        feat_extractor,
        os.path.join(frontend_dir, "campplus.onnx"),
        os.path.join(frontend_dir, "spk2info.pt"),
        DEVICE,
    )
    text_frontend = TextFrontEnd(use_phoneme)
    return frontend, text_frontend

class VllmRasLogitsProcessor(AdapterLogitsProcessor):
    """Approximate the original GLMTTS RAS behavior inside vLLM."""

    @classmethod
    def validate_params(cls, params: VllmSamplingParams):
        extra_args = params.extra_args or {}
        if not extra_args.get("ras_enabled", False):
            return None

        top_p = extra_args.get("ras_top_p", RAS_TOP_P)
        top_k = extra_args.get("ras_top_k", RAS_TOP_K)
        win_size = extra_args.get("ras_win_size", RAS_WIN_SIZE)
        tau_r = extra_args.get("ras_tau_r", RAS_TAU_R)
        temperature = extra_args.get("ras_temperature", RAS_TEMPERATURE)

        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"ras_top_p must be in (0, 1], got {top_p}")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"ras_top_k must be a positive integer, got {top_k}")
        if not isinstance(win_size, int) or win_size <= 0:
            raise ValueError(
                f"ras_win_size must be a positive integer, got {win_size}"
            )
        if tau_r < 0:
            raise ValueError(f"ras_tau_r must be >= 0, got {tau_r}")
        if temperature <= 0:
            raise ValueError(f"ras_temperature must be > 0, got {temperature}")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: VllmSamplingParams,
    ) -> RequestLogitsProcessor | None:
        extra_args = params.extra_args or {}
        if not extra_args.get("ras_enabled", False):
            return None

        top_p = float(extra_args.get("ras_top_p", RAS_TOP_P))
        top_k = int(extra_args.get("ras_top_k", RAS_TOP_K))
        win_size = int(extra_args.get("ras_win_size", RAS_WIN_SIZE))
        tau_r = float(extra_args.get("ras_tau_r", RAS_TAU_R))
        temperature = float(extra_args.get("ras_temperature", RAS_TEMPERATURE))
        repeat_threshold = max(1, math.ceil(win_size * tau_r))

        def ras_logits_processor(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            logits_fp32 = logits.float()
            base_probs = torch.softmax(logits_fp32, dim=-1)

            scaled_logits = logits_fp32 / temperature
            sorted_probs, sorted_idx = torch.softmax(scaled_logits, dim=-1).sort(
                descending=True, stable=True
            )

            selected_probs = []
            selected_indices = []
            cum_prob = 0.0
            for i in range(len(sorted_idx)):
                if cum_prob < top_p and len(selected_probs) < top_k:
                    prob_i = sorted_probs[i]
                    cum_prob += float(prob_i.item())
                    selected_probs.append(prob_i)
                    selected_indices.append(int(sorted_idx[i].item()))
                else:
                    break

            if not selected_probs:
                return logits

            selected_probs_t = torch.stack(selected_probs)
            selected_idx_t = torch.tensor(
                selected_indices, dtype=torch.long, device=logits.device
            )
            selected_probs_t = selected_probs_t / selected_probs_t.sum()

            repeated_mask = torch.zeros_like(selected_probs_t, dtype=torch.bool)
            if output_ids:
                recent_ids = output_ids[-win_size:]
                for idx, token_id in enumerate(selected_indices):
                    if recent_ids.count(token_id) >= repeat_threshold:
                        repeated_mask[idx] = True

            if repeated_mask.any():
                fallback_mass = selected_probs_t[repeated_mask].sum()
                final_probs = fallback_mass * base_probs
                final_probs[selected_idx_t[~repeated_mask]] += selected_probs_t[
                    ~repeated_mask
                ]
            else:
                final_probs = torch.zeros_like(base_probs)
                final_probs[selected_idx_t] = selected_probs_t

            final_probs = final_probs.clamp_min(torch.finfo(final_probs.dtype).tiny)
            return torch.log(final_probs).to(logits.dtype)

        return ras_logits_processor


def local_llm_forward(
    llm,
    prompt_text_token,
    tts_text_token,
    prompt_speech_token,
    beam_size=1,
    sampling=25,
    sample_method="ras",
    sampling_top_p=HF_SAMPLE_TOP_P,
    sampling_top_k=HF_SAMPLE_TOP_K,
    sampling_temperature=HF_SAMPLE_TEMPERATURE,
    seed=0,
):
    """
    Single LLM forward pass.
    """
    prompt_speech_token_offset = prompt_speech_token + llm.ats
    input_ids = (
        prompt_text_token.squeeze().tolist()
        + tts_text_token.squeeze().tolist()
        + [llm.boa]
        + prompt_speech_token_offset.squeeze().tolist()
    )

    text_len = tts_text_token.shape[1]
    use_ras = sample_method == "ras"
    allowed_token_ids = [llm.eoa] + list(range(llm.ats, llm.ate + 1))
    sampling_params = SamplingParams(
        temperature=RAS_TEMPERATURE if use_ras else sampling_temperature,
        top_p=1.0 if use_ras else 0.8,
        top_k=0 if use_ras else 25,
        seed=seed,
        max_tokens=int(text_len * 20),
        min_tokens=int(text_len * 2),
        stop_token_ids=[llm.eoa],
        ignore_eos=False,
        allowed_token_ids=allowed_token_ids,
        extra_args={
            "ras_enabled": use_ras,
            "ras_top_p": RAS_TOP_P,
            "ras_top_k": RAS_TOP_K,
            "ras_win_size": RAS_WIN_SIZE,
            "ras_tau_r": RAS_TAU_R,
            "ras_temperature": RAS_TEMPERATURE,
        },
    )

    engine = getattr(llm, "vllm_engine", None)
    if engine is None:
        engine = llm.llm.vllm_engine

    try:
        outputs = engine.generate(
            prompts=[input_ids],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
    except TypeError:
        outputs = engine.generate(
            inputs=[{"prompt_token_ids": input_ids}],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

    generated_ids = outputs[0].outputs[0].token_ids
    while generated_ids and generated_ids[-1] == llm.eoa:
        generated_ids = generated_ids[:-1]
    out_tokens = [t - llm.ats for t in generated_ids]

    return out_tokens


def local_flow_forward(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
    """
    Single Flow forward pass.
    """
    wav, full_mel = flow.token2wav_with_cache(
        token_list,
        n_timesteps=12,
        prompt_token=prompt_speech_tokens,
        prompt_feat=speech_feat,
        embedding=embedding,
    )
    return wav.detach().cpu(), full_mel


def get_prompt_cache_entry(
    frontend, text_frontend, prompt_text_raw, prompt_speech_path, sample_rate
):
    normalized_prompt_text = text_frontend.text_normalize(prompt_text_raw)
    prompt_cache_key = (
        normalized_prompt_text,
        os.path.abspath(prompt_speech_path),
        sample_rate,
    )

    if prompt_cache_key in PROMPT_CACHE:
        return PROMPT_CACHE[prompt_cache_key]

    prompt_text_token = frontend._extract_text_token(normalized_prompt_text + " ")
    prompt_speech_token = frontend._extract_speech_token([prompt_speech_path])
    speech_feat = frontend._extract_speech_feat(
        prompt_speech_path, sample_rate=sample_rate
    )
    embedding = frontend._extract_spk_embedding(prompt_speech_path)
    cache_speech_token = [prompt_speech_token.squeeze().tolist()]
    flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(DEVICE)

    prompt_entry = {
        "prompt_text": normalized_prompt_text,
        "prompt_text_token": prompt_text_token,
        "prompt_speech_token": prompt_speech_token,
        "speech_feat": speech_feat,
        "embedding": embedding,
        "cache_speech_token": cache_speech_token,
        "flow_prompt_token": flow_prompt_token,
    }
    PROMPT_CACHE[prompt_cache_key] = prompt_entry
    return prompt_entry


# --- Helper Function: Get Prompt from Cache ---
def get_cached_prompt(cache, synth_text_token, device=DEVICE):
    """
    Constructs prompt tokens from the cache.
    Prunes the cache if the sequence length exceeds MAX_LLM_SEQ_INP_LEN.
    """
    cache_text = cache["cache_text"]
    cache_text_token = cache["cache_text_token"]
    cache_speech_token = cache["cache_speech_token"]

    def __len_cache_text_token():
        return sum(map(lambda x: x.shape[1], cache_text_token))

    def __len_cache_speech_token():
        return sum(map(len, cache_speech_token))

    # Estimate required length ratio
    # Avoid division by zero
    text_len = __len_cache_text_token()
    ta_ratio = __len_cache_speech_token() / (text_len if text_len > 0 else 1.0)

    __len_synth_text_token = synth_text_token.shape[1]
    __len_synth_audi_token_estim = int(ta_ratio * __len_synth_text_token)

    # Prune cache if too long.
    # Logic: Keep the first item (original prompt), remove from the second item onwards.
    while (
        __len_cache_speech_token() + __len_synth_audi_token_estim
        > MAX_LLM_SEQ_INP_LEN
    ):
        if len(cache_speech_token) <= 1:
            break
        cache_text.pop(1)
        cache_text_token.pop(1)
        cache_speech_token.pop(1)

    # Construct Text Prompt
    prompt_text_token_from_cache = []
    for a_token in cache_text_token:
        prompt_text_token_from_cache.extend(a_token.squeeze().tolist())

    prompt_text_token = torch.tensor([prompt_text_token_from_cache]).to(device)

    # Construct Speech Prompt
    speech_tokens = []
    for a_cache_speech_token in cache_speech_token:
        speech_tokens.extend(a_cache_speech_token)

    llm_speech_token = torch.tensor([speech_tokens], dtype=torch.int32).to(device)

    return prompt_text_token, llm_speech_token


# --- Main Generation Logic ---
def generate_long(
    frontend: TTSFrontEnd,
    text_frontend: TextFrontEnd,
    llm,
    flow,
    text_info,
    cache,
    device,
    embedding,
    seed=0,
    sample_method="ras",
    sampling_top_p=HF_SAMPLE_TOP_P,
    sampling_top_k=HF_SAMPLE_TOP_K,
    sampling_temperature=HF_SAMPLE_TEMPERATURE,
    dump_tokens=False,
    token_dump_dir=None,
    flow_prompt_token=None,
    speech_feat=None,
    local_llm_forward=local_llm_forward,
    local_flow_forward=local_flow_forward,
    use_phoneme=False,
):
    outputs = []
    full_mels = []
    output_token_list = []
    uttid = text_info[0]
    syn_text = text_info[1]
    text_tn_dict = {
        "uttid": uttid,
        "syn_text": syn_text,
        "syn_text_tn": [],
        "syn_text_phoneme": [],
    }
    chunk_records = []
    short_text_list = text_frontend.split_by_len(syn_text)
    logging.info(f"[generate_long] uttid={uttid} split into {len(short_text_list)} chunks")

    for chunk_idx, tts_text in enumerate(short_text_list):
        chunk_start = time.perf_counter()
        logging.info(
            f"[generate_long] uttid={uttid} chunk={chunk_idx + 1}/{len(short_text_list)} raw_chars={len(tts_text)}"
        )
        seed_util.set_seed(seed)
        normalize_start = time.perf_counter()
        tts_text_tn = text_frontend.text_normalize(
            tts_text
        )  # Normalize again after splitting
        text_tn_dict["syn_text_tn"].append(tts_text_tn)
        if use_phoneme:
            tts_text_tn = text_frontend.g2p_infer(tts_text_tn)
            text_tn_dict["syn_text_phoneme"].append(tts_text_tn)
        normalize_elapsed = _elapsed_seconds(normalize_start)

        text_token_start = time.perf_counter()
        tts_text_token = frontend._extract_text_token(tts_text_tn)
        text_token_elapsed = _elapsed_seconds(text_token_start)

        # Access cache references
        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]

        # Determine Prompts
        prompt_prepare_start = time.perf_counter()
        if cache["use_cache"] and len(cache_text_token) > 1:
            prompt_text_token, prompt_speech_token = get_cached_prompt(
                cache, tts_text_token, device
            )
        else:
            prompt_text_token = cache_text_token[0].to(device)
            prompt_speech_token = torch.tensor(
                [cache_speech_token[0]], dtype=torch.int32
            ).to(device)
            logging.debug("[generate_long] Using initial prompt (empty cache history)")
        prompt_prepare_elapsed = _elapsed_seconds(prompt_prepare_start)

        # LLM Inference
        llm_start = time.perf_counter()
        token_list_res = local_llm_forward(
            llm=llm,
            prompt_text_token=prompt_text_token,
            tts_text_token=tts_text_token,
            prompt_speech_token=prompt_speech_token,
            sample_method=sample_method,
            sampling_top_p=sampling_top_p,
            sampling_top_k=sampling_top_k,
            sampling_temperature=sampling_temperature,
            seed=seed,
        )
        llm_elapsed = _elapsed_seconds(llm_start)

        output_token_list.extend(token_list_res)
        if dump_tokens:
            chunk_records.append(
                {
                    "chunk_idx": chunk_idx,
                    "raw_text": tts_text,
                    "normalized_text": tts_text_tn,
                    "num_tokens": len(token_list_res),
                    "tokens": token_list_res,
                }
            )

        # Flow Inference
        flow_start = time.perf_counter()
        output, full_mel = local_flow_forward(
            flow=flow,
            token_list=token_list_res,
            prompt_speech_tokens=flow_prompt_token,
            speech_feat=speech_feat,
            embedding=embedding,
        )
        flow_elapsed = _elapsed_seconds(flow_start)

        # Update Cache
        if cache is not None:
            cache_text.append(tts_text_tn)
            cache_text_token.append(tts_text_token)
            cache_speech_token.append(token_list_res)

        outputs.append(output)
        if full_mel is not None:
            full_mels.append(full_mel)

        logging.info(
            f"[generate_long] uttid={uttid} chunk={chunk_idx + 1}/{len(short_text_list)} timings: normalize={normalize_elapsed:.3f}s text_token={text_token_elapsed:.3f}s prompt={prompt_prepare_elapsed:.3f}s llm={llm_elapsed:.3f}s flow={flow_elapsed:.3f}s total={_elapsed_seconds(chunk_start):.3f}s out_tokens={len(token_list_res)} out_samples={output.shape[-1]}"
        )

    tts_speech = torch.concat(outputs, dim=1)
    tts_mel = torch.concat(full_mels, dim=-1) if full_mels else None
    if dump_tokens:
        _write_token_dump(
            dump_dir=token_dump_dir,
            uttid=uttid,
            backend=getattr(llm, "inference_backend", "vllm"),
            sample_method=sample_method,
            chunk_records=chunk_records,
        )

    return tts_speech, tts_mel, output_token_list, text_tn_dict


def jsonl_generate(
    data_name,
    folder_path,
    sample_rate=24000,
    seed=0,
    use_cache=True,
    use_phoneme=False,
    sample_method="ras",
    sampling_top_p=HF_SAMPLE_TOP_P,
    sampling_top_k=HF_SAMPLE_TOP_K,
    sampling_temperature=HF_SAMPLE_TEMPERATURE,
    dump_tokens=False,
    token_dump_dir=None,
):
    jsonl_path = os.path.join("examples", data_name + ".jsonl")

    # Dataset path resolution
    logging.info(f"Using jsonl: {jsonl_path}")
    item_list = file_utils.get_jsonl(jsonl_path)
    logging.info(f"[jsonl_generate] loaded {len(item_list)} items from {jsonl_path}")

    output_json_path = os.path.join(folder_path, "text_compare.jsonl")

    with open(output_json_path, "w") as f_out:
        for item_idx, item in enumerate(tqdm.tqdm(item_list), start=1):
            try:
                item_start = time.perf_counter()
                uttid = item["uttid"]
                wav_save_path = os.path.join(folder_path, f"{uttid}.wav")

                # Text Normalization
                prompt_prepare_start = time.perf_counter()
                prompt_entry = get_prompt_cache_entry(
                    frontend=frontend,
                    text_frontend=text_frontend,
                    prompt_text_raw=item["prompt_text"],
                    prompt_speech_path=item["prompt_speech"],
                    sample_rate=sample_rate,
                )
                prompt_text = prompt_entry["prompt_text"]
                prompt_text_token = prompt_entry["prompt_text_token"]
                prompt_speech_token = prompt_entry["prompt_speech_token"]
                speech_feat = prompt_entry["speech_feat"]
                embedding = prompt_entry["embedding"]
                cache_speech_token = [
                    tokens.copy() for tokens in prompt_entry["cache_speech_token"]
                ]
                flow_prompt_token = prompt_entry["flow_prompt_token"]
                synth_text = text_frontend.text_normalize(item["syn_text"])
                prompt_prepare_elapsed = _elapsed_seconds(prompt_prepare_start)

                cache = {
                    "cache_text": [prompt_text],
                    "cache_text_token": [prompt_text_token],
                    "cache_speech_token": cache_speech_token,
                    "use_cache": use_cache,
                }
                syn_text = item["syn_text"]
                logging.info(
                    f"[jsonl_generate] item={item_idx}/{len(item_list)} uttid={uttid} prompt_prep={prompt_prepare_elapsed:.3f}s syn_chars={len(syn_text)}"
                )

                # Run Generation
                generate_start = time.perf_counter()
                tts_speech, _, _, text_tn_dict = generate_long(
                    frontend=frontend,
                    text_frontend=text_frontend,
                    llm=llm,
                    flow=flow,
                    text_info=[uttid, synth_text],
                    cache=cache,
                    embedding=embedding,
                    seed=seed,
                    sample_method=sample_method,
                    sampling_top_p=sampling_top_p,
                    sampling_top_k=sampling_top_k,
                    sampling_temperature=sampling_temperature,
                    dump_tokens=dump_tokens,
                    token_dump_dir=token_dump_dir,
                    flow_prompt_token=flow_prompt_token,
                    speech_feat=speech_feat,
                    device=DEVICE,
                    use_phoneme=use_phoneme,
                )
                generate_elapsed = _elapsed_seconds(generate_start)

                f_out.write(
                    json.dumps(text_tn_dict, ensure_ascii=False, indent=2) + "\n"
                )
                f_out.flush()

                # Save Wave and Tokens
                save_start = time.perf_counter()
                os.makedirs(os.path.dirname(wav_save_path), exist_ok=True)
                import scipy.io.wavfile as wavfile

                speech_np = tts_speech.to("cpu").numpy().squeeze()
                wavfile.write(wav_save_path, sample_rate, speech_np)
                save_elapsed = _elapsed_seconds(save_start)

                audio_seconds = tts_speech.shape[-1] / sample_rate
                total_elapsed = _elapsed_seconds(item_start)
                rtf = total_elapsed / audio_seconds if audio_seconds > 0 else float("inf")
                logging.info(
                    f"[jsonl_generate] item={item_idx}/{len(item_list)} uttid={uttid} done prompt_prep={prompt_prepare_elapsed:.3f}s generate={generate_elapsed:.3f}s save={save_elapsed:.3f}s total={total_elapsed:.3f}s audio={audio_seconds:.3f}s rtf={rtf:.4f} wav={wav_save_path}"
                )
            except Exception as e:
                logging.error(f"Error processing {item.get('uttid', 'unknown')}: {e}")
                import traceback

                traceback.print_exc()

def load_models(
    use_phoneme=False,
    sample_rate=24000,
    llm_dtype="bf16",
):
    load_models_start = time.perf_counter()

    # Load Speech Tokenizer
    speech_tokenizer_path = os.path.join("ckpt", "speech_tokenizer")
    speech_tokenizer_start = time.perf_counter()
    _model, _feature_extractor = yaml_util.load_speech_tokenizer(
        speech_tokenizer_path
    )
    speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)
    logging.info(
        f"[load_models] speech_tokenizer={_elapsed_seconds(speech_tokenizer_start):.3f}s"
    )

    # Load Frontends
    frontend_start = time.perf_counter()
    frontend, text_frontend = load_frontends(
        speech_tokenizer, sample_rate=sample_rate, use_phoneme=use_phoneme
    )
    logging.info(f"[load_models] frontends={_elapsed_seconds(frontend_start):.3f}s")

    llama_path = os.path.abspath(os.path.join("ckpt", "llm"))
    tokenizer_path = os.path.abspath(os.path.join("ckpt", "vq32k-phoneme-tokenizer"))

    llm_start = time.perf_counter()
    llm = GLMTTS(llama_cfg_path=os.path.join(llama_path, "config.json"), mode="PRETRAIN")
    llm.inference_backend = "vllm"

    logging.info("🚀 Initializing vLLM Engine")
    logging.info(f"📂 Model: {llama_path}")
    logging.info(f"🎫 Tokenizer: {tokenizer_path}")

    llm.vllm_engine = LLM(
        model=llama_path,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
        enforce_eager=False,
        max_model_len=1024,
        block_size=128,
        disable_log_stats=True,
        logits_processors=[VllmRasLogitsProcessor],
    )
    llm.llama = None
    logging.info("✅ vLLM Engine is Ready!")

    special_token_ids = get_special_token_ids(frontend.tokenize_fn)
    llm.set_runtime_vars(special_token_ids=special_token_ids)

    flow_ckpt = os.path.join("ckpt", "flow", "flow.pt")
    flow_config = os.path.join("ckpt", "flow", "config.yaml")
    flow_start = time.perf_counter()
    flow = yaml_util.load_flow_model(flow_ckpt, flow_config, DEVICE)
    logging.info(f"[load_models] flow={_elapsed_seconds(flow_start):.3f}s")

    token2wav_start = time.perf_counter()
    token2wav = tts_model_util.Token2Wav(flow, sample_rate=sample_rate, device=DEVICE)
    logging.info(
        f"[load_models] token2wav={_elapsed_seconds(token2wav_start):.3f}s total={_elapsed_seconds(load_models_start):.3f}s"
    )

    return frontend, text_frontend, speech_tokenizer, llm, token2wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GLM-TTS Inference Script (Pretrain Mode Only)"
    )
    parser.add_argument("--data", default="example_zh", type=str)
    parser.add_argument("--exp_name", default="_test", type=str)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--use_phoneme", action="store_true", default=False)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument(
        "--llm_dtype",
        default="bf16",
        choices=["fp32", "bf16", "fp16", "int8"],
        help="LLM dtype",
    )
    parser.add_argument("--sample_method", default="ras", choices=["ras", "topk", "nucleus"])
    parser.add_argument("--sampling_top_p", type=float, default=HF_SAMPLE_TOP_P)
    parser.add_argument("--sampling_top_k", type=int, default=HF_SAMPLE_TOP_K)
    parser.add_argument("--sampling_temperature", type=float, default=HF_SAMPLE_TEMPERATURE)
    parser.add_argument("--dump_tokens", action="store_true", default=False)
    parser.add_argument("--token_dump_dir", default=None, type=str)

    args = parser.parse_args()

    # Load Models
    main_start = time.perf_counter()
    frontend, text_frontend, speech_tokenizer, llm, flow = load_models(
        use_phoneme=args.use_phoneme,
        sample_rate=args.sample_rate,
        llm_dtype=args.llm_dtype,
    )

    # Create Output Directory
    folder_path = os.path.join(
        CURRENT_DIR, "outputs", f"pretrain{args.exp_name}", args.data
    )
    os.makedirs(folder_path, exist_ok=True)
    logging.info(f"Output folder: {folder_path}")

    # Run Inference
    jsonl_generate(
        args.data,
        folder_path,
        sample_rate=args.sample_rate,
        use_cache=args.use_cache,
        use_phoneme=args.use_phoneme,
        sample_method=args.sample_method,
        sampling_top_p=args.sampling_top_p,
        sampling_top_k=args.sampling_top_k,
        sampling_temperature=args.sampling_temperature,
        dump_tokens=args.dump_tokens,
        token_dump_dir=args.token_dump_dir,
    )
    logging.info(f"[main] finished total={_elapsed_seconds(main_start):.3f}s")
