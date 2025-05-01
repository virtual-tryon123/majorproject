from datasets import load_from_disk
from torchvision import transforms
from PIL import Image

# Load dataset
dataset = load_from_disk("deepfashion_dataset")

# Define image transformations
image_transforms = transforms.Compose([
    transforms.Resize(512, interpolation=Image.BILINEAR),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Preprocess function
def preprocess(example):
    image = example["image"]
    if isinstance(image, Image.Image):
        example["pixel_values"] = image_transforms(image)
    else:
        example["pixel_values"] = image_transforms(Image.open(image))
    
    # Rename text column if needed
    example["input_ids"] = example["text"]  # Actual tokenization will be handled by the trainer
    return example

# Apply preprocessing
dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

