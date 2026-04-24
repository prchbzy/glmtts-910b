# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
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
"""
Flow Matching Wrapper for Streaming Inference.
Manages the DiT model initialization and the ODE sampling process.
"""

import torch
from torch.nn import functional as F
from cosyvoice.utils.mask import make_pad_mask
from flow.dit import DiT


class Flow(torch.nn.Module):
    def __init__(self,
                 # Speech token configuration
                 speech_token_dim: int = 512,
                 vocab_size: int = 100000,
                 input_frame_rate: float = 12.5,
                 # Speaker embedding configuration
                 spk_embed_dim: int = 80,
                 mel_dim: int = 80,
                 big_model: bool = False,
                 spkr_emb_adaLN: bool = False,
                 # Flags and options
                 calc_prompt_mel_loss: bool = True,
                 speech_token_cfg: bool = True,
                 use_wavlm_emb: bool = False,
                 remove_spkr_concat_condition: bool = False,
                 t_scheduler: str = 'cosine',
                 loss_type: str = "l2",
                 use_mdt: bool = False,
                 mel_framerate: int = 86,
                 ):
        super().__init__()
        self.mel_dim = mel_dim
        self.input_frame_rate = input_frame_rate
        self.mel_framerate = mel_framerate
        
        # Configuration Flags
        self.calc_prompt_mel_loss = calc_prompt_mel_loss
        self.use_wavlm_emb = use_wavlm_emb
        self.remove_spkr_concat_condition = remove_spkr_concat_condition
        self.spkr_emb_adaLN = spkr_emb_adaLN
        self.speech_token_cfg = speech_token_cfg
        self.loss_type = loss_type
        self.use_mdt = use_mdt

        # Scheduler and CFG settings
        self.t_scheduler = t_scheduler
        self.training_cfg_rate = 0.2
        self.inference_cfg_rate = 0.7
        self.sigma_min = 1e-06

        # Initialize Speaker Embedding Layer (Project to match DiT condition dim)
        if not remove_spkr_concat_condition:
            # 192 is the expected input dimension of the raw speaker embedding
            self.spk_embed_affine_layer = torch.nn.Linear(192, spk_embed_dim)

        # DiT Model Configuration
        dit_config = {
            "trans_dim": 1024 if big_model else 768,
            "depth": 22 if big_model else 18,
            "heads": 16 if big_model else 12,
            "ff_mult": 2,
            "text_emb_dim": speech_token_dim,
            "conv_layers": 4,
            "mel_dim": mel_dim,
            "text_vocab_size": vocab_size,
            "condition_dim": mel_dim if remove_spkr_concat_condition else mel_dim + spk_embed_dim,
            "spkr_emb_adaLN": spkr_emb_adaLN,
            "wav_lm_emb": use_wavlm_emb,
        }

        # Use 'model' instead of 'estimator' for clarity, though estimator is fine
        self.estimator = DiT(**dit_config)

    @torch.inference_mode()
    def inference_with_cache(self,
                             token,
                             prompt_token,
                             prompt_feat,
                             embedding,
                             n_timesteps=10,
                             last_step_cache=None,
                             wavlm_emb_bt=None,
                             is_causal=False,
                             block_pattern=None
                             ):
        """
        Streaming inference method that supports KV-caching via last_step_cache.
        """
        assert token.shape[0] == 1, "Batch size must be 1 for streaming inference."
        device = token.device
        token_len = token.shape[1]

        # Handle Prompt Concatenation
        if prompt_token is not None and prompt_feat is not None:
            token = torch.cat([prompt_token, token], dim=1)
            prompt_token_len = prompt_token.shape[1]
            token_len += prompt_token_len
            prompt_feat_len = prompt_feat.shape[1]
        else:
            prompt_feat_len = 0

        # Calculate feature length based on frame rates
        feat_len = int(token_len / self.input_frame_rate * self.mel_framerate)

        # Prepare masks and conditions
        padding_mask_bt = (~make_pad_mask(torch.LongTensor([feat_len]))).to(device)
        mel_cond_btd = torch.zeros([1, feat_len, self.mel_dim]).to(device)

        # Align prompt features
        if prompt_feat is not None:
            # Ensure we don't exceed dimensions when copying prompt features
            copy_len = min(prompt_feat.shape[1], mel_cond_btd.shape[1])
            mel_cond_btd[:, :copy_len, :] = prompt_feat[:, -copy_len:, :]

        # Process Speaker Embedding
        spkr_embedding_normed = F.normalize(embedding, dim=1)
        spk_proj_dtype = self.spk_embed_affine_layer.weight.dtype if not self.remove_spkr_concat_condition else spkr_embedding_normed.dtype

        if not self.remove_spkr_concat_condition:
            spkr_embedding = self.spk_embed_affine_layer(spkr_embedding_normed.to(dtype=spk_proj_dtype))
            # Expand speaker embedding to match time dimension
            spkr_embedding_expanded = spkr_embedding.unsqueeze(1).expand(-1, mel_cond_btd.shape[1], -1)
            # Concatenate mel-condition and speaker-condition
            condition_btd = torch.cat([mel_cond_btd, spkr_embedding_expanded], dim=-1)
        else:
            condition_btd = mel_cond_btd

        # Handle WavLM Embedding
        if self.use_wavlm_emb and wavlm_emb_bt is not None:
            emb2 = F.normalize(wavlm_emb_bt.to(device), dim=1)
            spkr_embedding_normed = torch.cat([spkr_embedding_normed, emb2], dim=1)

        # Run Sampling
        result_btd, current_step2cache = self.do_sample(
            token,
            mel_cond_btd,
            condition_btd, 
            padding_mask_bt, 
            spkr_embedding_normed, 
            is_causal,
            block_pattern, 
            n_timesteps, 
            last_step_cache
        )

        # Remove the prompt part from the result
        if prompt_feat is not None:
            result_btd = result_btd[:, prompt_feat_len:, :]

        # Permute to (Batch, Dimension, Time) for output convention
        result_bdt = result_btd.permute(0, 2, 1)

        return result_bdt, current_step2cache

    def do_sample(self, 
                  speech_token_bt,
                  mel_cond_btd,
                  condition_btd, 
                  padding_mask_bt, 
                  spkr_embedding_normed, 
                  is_causal,
                  block_pattern, 
                  n_timesteps, 
                  last_step_cache):
        """
        Executes the sampling process using a manual Euler method loop to support
        step-by-step caching for streaming.
        """
        current_step2cache = {}

        estimator_dtype = next(self.estimator.parameters()).dtype
        x = torch.randn_like(mel_cond_btd, dtype=estimator_dtype)
        condition_btd = condition_btd.to(dtype=estimator_dtype)
        spkr_embedding_normed = spkr_embedding_normed.to(dtype=estimator_dtype)
        device = speech_token_bt.device
        om_manager = getattr(self, 'om_estimator_manager', None)
        om_manager_initialized = getattr(self, 'om_estimator_manager_initialized', False)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        t_current = t_span[0]
        dt = t_span[1] - t_span[0]

        seq_len = x.shape[1]
        om_text_embed = None
        om_text_embed_uncond = None
        if not om_manager_initialized:
            try:
                om_text_embed = self.estimator.text_emb_layer(speech_token_bt, seq_len)
                if self.inference_cfg_rate > 0 and self.speech_token_cfg:
                    text_for_no_condition = torch.zeros_like(speech_token_bt)
                    om_text_embed_uncond = self.estimator.text_emb_layer(text_for_no_condition, seq_len)
            except Exception as prep_err:
                print(f'[flow_om] text embed precompute failed before om init: {prep_err}')

        if om_manager is None and not om_manager_initialized:
            from flow.om_runtime import maybe_create_flow_om_manager
            om_manager = maybe_create_flow_om_manager()
            self.om_estimator_manager = om_manager
            self.om_estimator_manager_initialized = True

        if om_manager is not None:
            try:
                chosen_bucket = om_manager.select_bucket(seq_len)
                print(
                    f'[flow_om] request seq_len={seq_len} bucket={chosen_bucket} '
                    f'timesteps={n_timesteps} cfg={self.inference_cfg_rate > 0}'
                )
                if om_text_embed is None:
                    om_text_embed = self.estimator.text_emb_layer(speech_token_bt, seq_len)
                    if self.inference_cfg_rate > 0 and self.speech_token_cfg:
                        text_for_no_condition = torch.zeros_like(speech_token_bt)
                        om_text_embed_uncond = self.estimator.text_emb_layer(text_for_no_condition, seq_len)
            except Exception as bucket_err:
                print(f'[flow_om] prepare failed, fallback to PyTorch: {bucket_err}')
                om_manager = None

        for step in range(1, len(t_span)):
            if last_step_cache is not None:
                x_cache = last_step_cache[step]['x']
                override_len = last_step_cache.get('override_len', x_cache.shape[-1])
                safe_len = min(x.shape[1], override_len)
                x[:, :safe_len, :] = x_cache[:, :safe_len, :]

            current_step2cache[step] = {
                "x": x.clone().detach(),
            }

            if om_manager is not None:
                try:
                    if self.inference_cfg_rate > 0:
                        cond_out = om_manager.infer(
                            middle_point_btd=x,
                            condition_btd=condition_btd,
                            precomputed_text_embed=om_text_embed,
                            time_step_1d=t_current[None],
                            padding_mask_bt=padding_mask_bt,
                            spkr_emb_bd=spkr_embedding_normed,
                        )
                        uncond_text_embed = om_text_embed_uncond if self.speech_token_cfg else om_text_embed
                        cfg_out = om_manager.infer(
                            middle_point_btd=x,
                            condition_btd=torch.zeros_like(condition_btd),
                            precomputed_text_embed=uncond_text_embed,
                            time_step_1d=t_current[None],
                            padding_mask_bt=padding_mask_bt,
                            spkr_emb_bd=torch.zeros_like(spkr_embedding_normed),
                        )
                        dphi_dt = (
                            (1.0 + self.inference_cfg_rate) * cond_out
                            - self.inference_cfg_rate * cfg_out
                        )
                    else:
                        dphi_dt = om_manager.infer(
                            middle_point_btd=x,
                            condition_btd=condition_btd,
                            precomputed_text_embed=om_text_embed,
                            time_step_1d=t_current[None],
                            padding_mask_bt=padding_mask_bt,
                            spkr_emb_bd=spkr_embedding_normed,
                        )
                except Exception as om_err:
                    print(f'[flow_om] infer failed, fallback to PyTorch: {om_err}')
                    om_manager = None

            if om_manager is None:
                if self.inference_cfg_rate > 0:
                    if self.speech_token_cfg:
                        text_for_no_condition = torch.zeros_like(speech_token_bt)
                    else:
                        text_for_no_condition = speech_token_bt

                    x_batched = torch.cat([x, x], dim=0)
                    condition_batched = torch.cat(
                        [condition_btd, torch.zeros_like(condition_btd)],
                        dim=0,
                    )
                    text_batched = torch.cat(
                        [speech_token_bt, text_for_no_condition],
                        dim=0,
                    )
                    time_batched = t_current[None].repeat(2)
                    padding_mask_batched = torch.cat(
                        [padding_mask_bt, padding_mask_bt],
                        dim=0,
                    )
                    spkr_emb_batched = torch.cat(
                        [
                            spkr_embedding_normed,
                            torch.zeros_like(spkr_embedding_normed),
                        ],
                        dim=0,
                    )

                    dphi_all = self.estimator(
                        middle_point_btd=x_batched,
                        condition_btd=condition_batched,
                        text=text_batched,
                        time_step_1d=time_batched,
                        padding_mask_bt=padding_mask_batched,
                        spkr_emb_bd=spkr_emb_batched,
                        is_causal=is_causal,
                        block_pattern=block_pattern,
                    )

                    dphi_dt, cfg_dphi_dt = dphi_all.chunk(2, dim=0)
                    dphi_dt = (
                        (1.0 + self.inference_cfg_rate) * dphi_dt
                        - self.inference_cfg_rate * cfg_dphi_dt
                    )
                else:
                    dphi_dt = self.estimator(
                        middle_point_btd=x,
                        condition_btd=condition_btd,
                        text=speech_token_bt,
                        time_step_1d=t_current[None],
                        padding_mask_bt=padding_mask_bt,
                        spkr_emb_bd=spkr_embedding_normed,
                        is_causal=is_causal,
                        block_pattern=block_pattern,
                    )

            x = x + dt * dphi_dt
            t_current = t_current + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_current

        result_btd = x.float()
        return result_btd, current_step2cache
