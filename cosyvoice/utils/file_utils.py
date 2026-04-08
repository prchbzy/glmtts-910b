# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import torchaudio
import soundfile as sf
import torch
import torchaudio.transforms as T

def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists

def read_json_lists(list_file):
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, 'r', encoding='utf8') as fin:
            results.update(json.load(fin))
    return results

def load_wav(wav, target_sample_rate):
    data, sr = sf.read(wav)
    speech = torch.from_numpy(data).float()
    
    if speech.ndim == 1:
        speech = speech.unsqueeze(0)
    elif speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)

    if sr != target_sample_rate:
        resampler = T.Resample(sr, target_sample_rate)
        speech = resampler(speech)
    return speech, target_sample_rate

def speed_change(waveform, sample_rate, speed_factor: str):
    effects = [
        ["tempo", speed_factor],  # speed_factor
        ["rate", f"{sample_rate}"]
    ]
    augmented_waveform, new_sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform,
        sample_rate,
        effects
    )
    return augmented_waveform, new_sample_rate
