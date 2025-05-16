# run_vit_best_model.py: Evaluate accuracy of fine-tuned ViT model on Tiny-ImageNet
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from tqdm import tqdm
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Tiny-ImageNet dataset
    print("Loading Tiny-ImageNet...")
    dataset = load_dataset("zh-plus/tiny-imagenet", split="train[:200]")  # Use first 200 for consistency

    # 2. Load processor and model
    model_name = "WinKawaks/vit-tiny-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=200,
        ignore_mismatched_sizes=True  # 👈 this skips the classifier size mismatch
    )

    
    # 3. Load your trained weights
    print("Loading fine-tuned weights from best_model.pt")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    # 4. Loop over samples
    for sample in tqdm(dataset, desc="Evaluating"):
        img = sample["image"]
        label = sample["label"]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=-1).item()

        if pred == label:
            correct += 1
        total += 1

    acc = 100 * correct / total
    print(f"\nAccuracy over {total} samples: {acc:.2f}%")

if __name__ == "__main__":
    main()
