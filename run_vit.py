# run_vit.py: Example script to run inference with ViT
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

def main():
    # 1. Load the CIFAR-100 dataset from the Hugging Face Hub.
    # Note: Adjust the dataset identifier if needed.
    print("starting to load dataset")
    dataset = load_dataset("uoft-cs/cifar100")
    print("Dataset loaded with splits:", list(dataset.keys()))
    
    # 2. Get a sample from the training split.
    sample = dataset["train"][0]
    
    # Assume the image is stored under the key "img".
    img = sample["img"]
    # If it is not a PIL Image, convert it.
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    # 3. Load the image processor and pre-trained ViT model.
    # Here we use ViTImageProcessor (recommended over deprecated ViTFeatureExtractor)
    # and a ViT model from Hugging Face.
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    # 4. Preprocess the image.
    inputs = image_processor(images=img, return_tensors="pt")
    
    # 5. Run inference.
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label.get(predicted_idx, "Unknown")
    
    print("Predicted class:", predicted_label)
