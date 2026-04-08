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
import logging
import os

import gradio as gr

try:
    from tools.inference_service import clear_memory, run_gradio_inference
except ImportError:
    from inference_service import clear_memory, run_gradio_inference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_inference(prompt_text, prompt_audio_path, input_text, seed, sample_rate, use_cache=True):
    if not input_text:
        raise gr.Error("Please provide text to synthesize.")
    if not prompt_audio_path:
        raise gr.Error("Please upload a prompt audio file.")
    if not prompt_text:
        gr.Warning("Prompt text is empty. Results might be suboptimal.")

    try:
        return run_gradio_inference(
            prompt_text=prompt_text,
            prompt_audio_path=prompt_audio_path,
            input_text=input_text,
            seed=seed,
            sample_rate=sample_rate,
            use_cache=use_cache,
        )
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Inference failed: {str(e)}")

# --- Gradio UI Layout ---

with gr.Blocks(title="GLMTTS Inference") as app:
    gr.Markdown("# 🎵 GLMTTS Open Source Demo")
    gr.Markdown("Zero-shot text-to-speech generation using GLMTTS models.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Zero-Shot Prompt Settings")
            
            prompt_audio = gr.Audio(
                label="Upload Prompt Audio (Reference Voice)",
                type="filepath",
                value=os.path.join("examples", "prompt", "jiayan_zh.wav")
            )
            
            prompt_text = gr.Textbox(
                label="Prompt Text",
                placeholder="Enter the exact text spoken in the prompt audio...",
                lines=2,
                info="Accurate prompt text improves speaker similarity.",
                value="他当时还跟线下其他的站姐吵架，然后，打架进局子了。"
            )

            gr.Markdown("### 2. Input Settings")
            input_text = gr.Textbox(
                label="Text to Synthesize",
                value="我最爱吃人参果，你喜欢吃吗？", 
                lines=5
            )
            
            with gr.Accordion("Advanced Settings", open=True):
                # Update: Added Sample Rate selection
                sample_rate = gr.Radio(
                    choices=[24000, 32000], 
                    value=24000, 
                    label="Sample Rate (Hz)",
                    info="Choose 32000 for higher quality if model supports it."
                )
                seed = gr.Number(label="Seed", value=42, precision=0)
                use_cache = gr.Checkbox(label="Use KV Cache", value=True, info="Faster generation for long text.")

            generate_btn = gr.Button("🚀 Generate Audio", variant="primary", size="lg")
            clear_btn = gr.Button("🧹 Clear VRAM", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### 3. Output")
            output_audio = gr.Audio(label="Synthesized Result")
            status_msg = gr.Textbox(label="System Status", interactive=False)

    # Event Bindings
    generate_btn.click(
        fn=run_inference,
        inputs=[prompt_text, prompt_audio, input_text, seed, sample_rate, use_cache],
        outputs=[output_audio, status_msg]
    )

    clear_btn.click(
        fn=clear_memory,
        inputs=None,
        outputs=[status_msg]
    )

if __name__ == "__main__":
    app.queue().launch(
        server_name="0.0.0.0", 
        server_port=8048, 
        theme=gr.themes.Soft(),
        share=False
    )