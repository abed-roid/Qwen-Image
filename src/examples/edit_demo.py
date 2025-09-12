import gradio as gr
import numpy as np
import random
import torch
import spaces
import os
import base64
import json
import shutil
import time

from PIL import Image

# ---- Diffusers / Transformers (both variants) ----
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel

# =========================
# Shared: Prompt Polisher
# =========================
SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.
- All added objects or modifications must align with the logic and style of the edited input image’s overall scene.

## 2. Task Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:
    > Original: "Add an animal"
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Do not translate or alter the original language of the text, and do not change the capitalization.
- **For text replacement tasks, always use the fixed template:**
    - `Replace "xx" to "yy"`.
    - `Replace the xx bounding box to "yy"`.
- If the user does not specify text content, infer and add concise text based on the instruction and the input image’s context. For example:
    > Original: "Add a line of text" (poster)
    > Rewritten: "Add text "LIMITED EDITION" at the top center with slight shadow`
- Specify text position, color, and layout in a concise way.

### 3. Human Editing Tasks
- Maintain the person’s core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.).
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.
- **For expression changes, they must be natural and subtle, never exaggerated.**
- If deletion is not specifically emphasized, the most important subject in the original image (e.g., a person, an animal) should be preserved.
    - For background change tasks, emphasize maintaining subject consistency at first.
- Example:
    > Original: "Change the person’s hat"
    > Rewritten: "Replace the man’s hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged"

### 4. Style Transformation or Enhancement Tasks
- If a style is specified, describe it concisely with key visual traits. For example:
    > Original: "Disco style"
    > Rewritten: "1970s disco: flashing lights, disco ball, mirrored walls, colorful tones"
- If the instruction says "use reference style" or "keep current style," analyze the input image, extract main features (color, composition, texture, lighting, art style), and integrate them into the prompt.
- **For coloring tasks, including restoring old photos, always use the fixed template:** "Restore old photograph, remove scratches, reduce noise, enhance details, high resolution, realistic, natural skin tones, clear facial features, no distortion, vintage photo restoration"
- If there are other changes, place the style description at the end.

## 3. Rationality and Logic Checks
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" should be logically corrected.
- Add missing key information: if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edges).

# Output Format Example
```json
{
   "Rewritten": "..."
}
'''
def encode_image(pil_image):
    import io
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def api(prompt, img_list, model="qwen-vl-max-latest", kwargs={}):
    import dashscope
    api_key = os.environ.get('DASH_API_KEY')
    if not api_key:
        raise EnvironmentError("DASH_API_KEY is not set")
    assert model in ["qwen-vl-max-latest"], f"Not implemented model {model}"
    sys_promot = "you are a helpful assistant, you should provide useful answers to users."
    messages = [
        {"role": "system", "content": sys_promot},
        {"role": "user", "content": []}]
    for img in img_list:
        messages[1]["content"].append({"image": f"data:image/png;base64,{encode_image(img)}"})
    messages[1]["content"].append({"text": f"{prompt}"})

    response_format = kwargs.get('response_format', None)

    response = dashscope.MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=messages,
        result_format='message',
        response_format=response_format,
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content[0]['text']
    else:
        raise Exception(f'Failed to post: {response}')

def polish_prompt(prompt, img):
    prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"
    success = False
    while not success:
        try:
            result = api(prompt, [img])
            if isinstance(result, str):
                result = result.replace('```json','').replace('```','')
                result = json.loads(result)
            else:
                result = json.loads(result)
            polished_prompt = result['Rewritten'].strip().replace("\n", " ")
            success = True
        except Exception as e:
            print(f"[Warning] Error during API call: {e}")
    return polished_prompt

# =========================
# Global / Shared
# =========================
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = np.iinfo(np.int32).max

# =========================
# FULL generation (File 2)
# =========================
# Keep a single hot pipeline on GPU (as in File 2)

def infer_full(
    image: Image.Image,
    prompt: str,
    seed: int = 42,
    randomize_seed: bool = False,
    true_guidance_scale: float = 1.0,
    num_inference_steps: int = 50,
    width: int = 1024,
    height: int = 1024,
    ootd: float = 1.0,
    rewrite_prompt: bool = False,         # default locked false
    num_images_per_prompt: int = 1,       # default locked 1
    progress=gr.Progress(track_tqdm=True),
):
    """
    Mirrors File 2 behavior (full generation):
      - Uses a global QwenImageEditPipeline on device
      - Supports negative_prompt and true_cfg_scale
    """
    negative_prompt = " "

    pipe_full = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=dtype).to(device)
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    if rewrite_prompt:
        prompt = polish_prompt(prompt, image)
        print(f"[Full] Rewritten Prompt: {prompt}")

    print(f"[Full] Seed={seed}, Steps={num_inference_steps}, Guidance={true_guidance_scale}")

    images = pipe_full(
        image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=num_images_per_prompt
    ).images

    return images, seed

model_id = "Qwen/Qwen-Image-Edit"

# Quantized transformer (diffusers)
quant_config_diff = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
)
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quant_config_diff,
    torch_dtype=torch.bfloat16,
).to("cpu")

# Quantized text encoder (transformers)
quant_config_tx = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=quant_config_tx,
    torch_dtype=torch.bfloat16,
).to("cpu")

pipe = QwenImageEditPipeline.from_pretrained(
    model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16
)

pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors",
    adapter_name="lightning"
)
pipe.load_lora_weights(
    "/root/Qwen-Image/src/examples/ootd_colour-19-3600.safetensors",
    adapter_name="realism"
)

# Now set their influence
pipe.set_adapters(
    ["lightning", "realism"],
    adapter_weights=[1, 0.65]   # Lightning at 0.4, Realism at 0.7
)
pipe.enable_model_cpu_offload()
generator = torch.Generator(device="cuda").manual_seed(42)

def infer_fast(
    image: Image.Image,
    prompt: str,
    seed: int = 42,
    randomize_seed: bool = False,
    true_guidance_scale: float = 1.0,   # kept in signature for symmetry; not used
    num_inference_steps: int = 50,      # will be overridden to 8 (Lightning preset)
    width: int = 1024,
    height: int = 1024,
    ootd: float = 1.0,
    rewrite_prompt: bool = False,       # default locked false
    num_images_per_prompt: int = 1,     # locked to 1
    progress=gr.Progress(track_tqdm=True),
):
    """
    Mirrors File 1 behavior (fast generation):
      - Builds quantized pipeline + Lightning LoRA (8 steps)
      - Uses minimal call: pipe(image, prompt, num_inference_steps=8).images
      - Keeps negative prompt hardcoded to blank (File 1)
    """
    negative_prompt = " "
    dir_path = "/tmp/gradio"
    pipe.set_adapters(
        ["lightning", "realism"],
        adapter_weights=[1, ootd]   # Lightning at 0.4, Realism at 0.7
    )

    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        # Count only subdirectories
        folder_count = sum(1 for entry in os.scandir(dir_path) if entry.is_dir())

        if folder_count > 100:
            time.sleep(1)
            shutil.rmtree(dir_path)
            print(f"Deleted: {dir_path} (had {folder_count} folders)")
        else:
            print(f"Folder count = {folder_count}, no deletion needed.")
    else:
        print(f"Directory not found: {dir_path}")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    dvc = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=dvc).manual_seed(seed)

    if rewrite_prompt:
        prompt = polish_prompt(prompt, image)
        print(f"[Fast] Rewritten Prompt: {prompt}")

    print(f"[Fast] Seed={seed}, Steps=8 (Lightning), Guidance param ignored")

    # Force 8 steps for Lightning
    out = pipe(
        image, prompt, num_inference_steps=num_inference_steps, width=width, height=height
    ).images

    return out, seed

# =========================
# Dispatcher (routes by checkbox)
# =========================
def infer_dispatch(
    image,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=50,
    width=1024,
    height=1024,
    ootd=1.0,
    rewrite_prompt=False,            # locked default
    num_images_per_prompt=1,         # locked default
    fast=True,                       # locked default
    progress=gr.Progress(track_tqdm=True),
):
    if fast:
        return infer_fast(
            image, prompt, seed, randomize_seed, true_guidance_scale,
            num_inference_steps, width, height, ootd, rewrite_prompt, num_images_per_prompt
        )
    else:
        return infer_full(
            image, prompt, seed, randomize_seed, true_guidance_scale,
            num_inference_steps, width, height, ootd, rewrite_prompt, num_images_per_prompt
        )

# =========================
# UI
# =========================
examples = []

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML('<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Logo" width="400" style="display: block; margin: 0 auto;">')
        gr.Markdown("[Learn more](https://github.com/QwenLM/Qwen-Image) about the Qwen-Image series. Try on [Qwen Chat](https://chat.qwen.ai/), or [download model](https://huggingface.co/Qwen/Qwen-Image-Edit) to run locally with ComfyUI or diffusers.")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", show_label=False, type="pil")
            result = gr.Gallery(label="Result", show_label=False, type="pil")

        with gr.Row():
            prompt_in = gr.Text(
                label="Prompt",
                show_label=False,
                placeholder="describe the edit instruction",
                container=False,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            seed_in = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed_in = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                true_guidance_scale_in = gr.Slider(label="True guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=4.0)
                num_inference_steps_in = gr.Slider(label="Number of inference steps", minimum=1, maximum=24, step=1, value=8)
                # REMOVED: num_images_per_prompt_in (locked to 1)
                width = gr.Slider(label="Width", minimum=256, maximum=4096, step=1, value=1024)
                height = gr.Slider(label="Height", minimum=256, maximum=4096, step=1, value=1024)
                ootd = gr.Slider(label="OOTD", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                # REMOVED: rewrite_prompt_in (locked to False)

        # REMOVED: fast_checkbox UI (locked to True)

        # Hidden constants (States) to feed locked values
        rewrite_prompt_state = gr.State(False)
        num_images_per_prompt_state = gr.State(1)
        fast_state = gr.State(True)

        # Hidden buttons to expose explicit API endpoints
        api_only_fast = gr.Button(visible=False)
        api_only_full = gr.Button(visible=False)

    # Routed generation (UI + API) -> /api/predict/generate
    gr.on(
        triggers=[run_button.click, prompt_in.submit],
        fn=infer_dispatch,
        inputs=[
            input_image,
            prompt_in,
            seed_in,
            randomize_seed_in,
            true_guidance_scale_in,
            num_inference_steps_in,
            width,
            height,
            ootd,
            rewrite_prompt_state,          # locked False
            num_images_per_prompt_state,   # locked 1
            fast_state,                    # locked True
        ],
        outputs=[result, seed_in],
        api_name="generate"
    )

    # Explicit fast endpoint -> /api/predict/generate_fast
    api_only_fast.click(
        fn=infer_fast,
        inputs=[
            input_image,
            prompt_in,
            seed_in,
            randomize_seed_in,
            true_guidance_scale_in,
            num_inference_steps_in,
            width,
            height,
            ootd,
            rewrite_prompt_state,          # locked False
            num_images_per_prompt_state,   # locked 1
        ],
        outputs=[result, seed_in],
        api_name="generate_fast"
    )

    # Explicit full endpoint -> /api/predict/generate_full
    api_only_full.click(
        fn=infer_full,
        inputs=[
            input_image,
            prompt_in,
            seed_in,
            randomize_seed_in,
            true_guidance_scale_in,
            num_inference_steps_in,
            width,
            height,
            ootd,
            rewrite_prompt_state,          # locked False
            num_images_per_prompt_state,   # locked 1
        ],
        outputs=[result, seed_in],
        api_name="generate_full"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
