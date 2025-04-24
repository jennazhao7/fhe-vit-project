import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Ensure output dir exists
os.makedirs("input", exist_ok=True)

# Load model info to get normalization parameters
model_name = "WinKawaks/vit-tiny-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

# Define preprocessing
transform = Compose([
    Resize((224, 224)),  # Match ViT expected input size
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std),
])

# Load small subset
dataset = load_dataset("johnowhitaker/imagenette2-320", split="train[:5]")

# Save each image as .bin file
for i, sample in enumerate(dataset):
    image = sample["image"]
    label = sample["label"]

    img_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
    img_np = img_tensor.numpy().astype(np.float32)
    
    # Flatten and save to .bin file
    img_np.flatten().tofile(f"input/image_{i}.bin")

    # Save label too (optional, for ground truth logging)
    with open(f"input/label_{i}.txt", "w") as f:
        f.write(str(label))

    print(f"[✓] Saved input/image_{i}.bin and label_{i}.txt")