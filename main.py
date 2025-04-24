import os
import torch
import subprocess
import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm

model_name = "WinKawaks/vit-tiny-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name,use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = load_dataset("johnowhitaker/imagenette2-320", split="train[:5]")

transform = Compose([
    Resize((224,224)),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std),
])

correct = 0
total = 0

for i, sample in enumerate(tqdm(dataset)):
    image = sample["image"]
    label = sample["label"]
    print(f"[FHE] Ground Truth label: {label}")
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0) 
    img_np = img_tensor.numpy().astype(np.float32)
    img_np.tofile("input/image_input.bin")

    # Call FHE inference binary
    result = subprocess.run(["./build/fhe_inference"], capture_output=True, text=True)
    print(result.stdout.strip())
    lines = result.stdout.splitlines()
    pred_line = [line for line in lines if "[RESULT]Predicted label" in line]
    if pred_line:
        predicted = int(pred_line[0].split(":")[-1].strip())
        if predicted == label:
            correct += 1
        total += 1

print(f"\n✅ FHE Accuracy on {total} samples: {correct/total:.2%}")