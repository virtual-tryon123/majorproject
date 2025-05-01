from datasets import load_dataset, Dataset
from PIL import Image
import os

def load_deepfashion(image_dir, annotations):
    samples = []
    for img_name, caption in annotations.items():
        samples.append({
            "image": Image.open(os.path.join(image_dir, img_name)).convert("RGB"),
            "text": caption
        })
    return Dataset.from_list(samples)

dataset = load_deepfashion("data/images", {"img1.jpg": "Red evening dress with lace sleeves"})
dataset.save_to_disk("deepfashion_dataset")
