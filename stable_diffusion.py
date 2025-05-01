from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline
)
from transformers import CLIPTextModel, CLIPTokenizer
import torch

# Load VAE (for encoding/decoding images to latent space)
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="vae"
)

# Load UNet (core of the diffusion model, performs denoising)
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet"
)

# Load Text Encoder and Tokenizer (CLIP)
tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14"
)
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14"
)

# Load Scheduler (controls the noise schedule)
scheduler = DDPMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)

# Assemble into Stable Diffusion pipeline
pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,  # Optional: disable safety checker if needed
)

# Move to GPU (if available)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Example inference
prompt = "a stunning red dress on a mannequin"
image = pipe(prompt).images[0]

# Save the output
image.save("output_red_dress.png")
