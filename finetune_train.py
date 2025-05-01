from diffusers import StableDiffusionPipeline
from diffusers.training_utils import EMAModel
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available

from diffusers import StableDiffusionPipeline
from diffusers.training import StableDiffusionTrainingArguments, StableDiffusionTrainer

from datasets import load_from_disk

dataset = load_from_disk("deepfashion_dataset")

# Load base model
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

# Training arguments
args = StableDiffusionTrainingArguments(
    output_dir="./dress-generator-model",
    resolution=512,
    train_batch_size=1,
    num_train_epochs=5,
    learning_rate=1e-5,
    lr_scheduler="constant",
    gradient_accumulation_steps=2,
    mixed_precision="fp16",
    use_ema=True,
    enable_xformers_memory_efficient_attention=True,
    save_model_every_n_epochs=1,
    logging_dir="./logs",
    tracker_project_name="deepfashion-finetune"
)

trainer = StableDiffusionTrainer(
    model_name_or_path=pretrained_model_name_or_path,
    args=args,
    train_dataset=dataset,
)

trainer.train()
