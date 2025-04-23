# from transformers import AutoImageProcessor, ViTForImageClassification
# from PIL import Image
# import torch

# # Load local image
# image_path = "sample_image.png"
# image = Image.open(image_path).convert("RGB")  # Convert to RGB in case it's grayscale

# # Load model and processor
# model_name = "google/vit-base-patch16-224"
# processor = AutoImageProcessor.from_pretrained(model_name,use_fast=True)
# model = ViTForImageClassification.from_pretrained(model_name)

# # Preprocess
# inputs = processor(image, return_tensors="pt")

# # Inference
# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class_idx = logits.argmax(-1).item()

# # Output predicted label
# print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")

# import torch
# from transformers import ViTForImageClassification, AutoImageProcessor
# from datasets import load_dataset
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# # Device config
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model and processor
# model_name = "google/vit-base-patch16-224"
# model = ViTForImageClassification.from_pretrained(model_name).to(device)
# processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

# # Load dataset (CIFAR-10 test split)
# dataset = load_dataset("cifar10", split="test")

# # Define preprocessing compatible with ViT
# transform = Compose([
#     Resize((224, 224)),
#     ToTensor(),
#     Normalize(mean=processor.image_mean, std=processor.image_std)
# ])

# # Add transformed pixel values to dataset
# def preprocess(example):
#     # processor returns a dict with 'pixel_values': Tensor
#     processed = processor(example["img"], return_tensors="pt")
#     return {
#         "pixel_values": processed["pixel_values"].squeeze(0),  # Remove batch dim
#         "label": example["label"]
#     }

# dataset = dataset.map(preprocess)

# # Collate function for DataLoader
# def collate_fn(batch):
#     pixel_values = torch.stack([x["pixel_values"] for x in batch])
#     labels = torch.tensor([x["label"] for x in batch])
#     return {"pixel_values": pixel_values, "labels": labels}

# # DataLoader
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# # Run inference and compute accuracy
# correct = 0
# total = 0
# model.eval()

# with torch.no_grad():
#     for batch in tqdm(dataloader, desc="Evaluating"):
#         inputs = batch["pixel_values"].to(device)
#         labels = batch["labels"].to(device)

#         outputs = model(inputs)
#         preds = outputs.logits.argmax(dim=1)

#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

# accuracy = correct / total
# print(f"Accuracy on test set: {accuracy * 100:.2f}%")
# # Note: The above code assumes that the CIFAR-10 dataset is available and properly formatted.

# import torch
# from datasets import load_dataset
# from transformers import AutoImageProcessor, ViTForImageClassification
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from tqdm import tqdm

# # Load model & processor
# model_name = "nateraw/vit-base-patch16-224-cifar10"
# model = ViTForImageClassification.from_pretrained(model_name)
# processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
# model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Load test dataset
# dataset = load_dataset("cifar10", split="test")

# # Preprocessing (match ViT expected inputs)
# transform = Compose([
#     Resize((224, 224)),
#     ToTensor(),
#     Normalize(mean=processor.image_mean, std=processor.image_std)
# ])

# def preprocess(example):
#     image = example["img"].convert("RGB")  # enforce RGB
#     example["pixel_values"] = transform(image)
#     return example

# dataset = dataset.map(preprocess)

# # Run inference and compute accuracy
# correct = 0
# total = 0

# for example in tqdm(dataset, desc="Evaluating"):
#     inputs = example["pixel_values"].unsqueeze(0).to(device)

#     label = example["label"]

#     with torch.no_grad():
#         outputs = model(pixel_values=inputs)
#         pred = outputs.logits.argmax(-1).item()

#     if pred == label:
#         correct += 1
#     total += 1

# print(f"Accuracy: {100 * correct / total:.2f}%")

# import torch
# from datasets import load_dataset
# from transformers import AutoImageProcessor, ViTForImageClassification
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from PIL import Image

# # Configuration
# model_name = "nateraw/vit-base-patch16-224-cifar10"
# subset_size = 1000
# batch_size = 32
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model and processor
# model = ViTForImageClassification.from_pretrained(model_name)
# processor = AutoImageProcessor.from_pretrained(model_name)
# model.to(device)
# model.eval()

# # Load dataset subset
# dataset = load_dataset("cifar10", split="test[:1000]")

# # Preprocessing transform
# transform = Compose([
#     Resize((224, 224)),
#     ToTensor(),
#     Normalize(mean=processor.image_mean, std=processor.image_std)
# ])

# # Preprocess function
# def preprocess(example):
#     image = example["img"].convert("RGB")
#     tensor = transform(image)
#     return {"pixel_values": tensor, "label": example["label"]}

# dataset = dataset.map(preprocess)
# dataset.set_format(type="torch", columns=["pixel_values", "label"])

# # DataLoader for batching
# dataloader = DataLoader(dataset, batch_size=batch_size)

# # Inference loop
# correct = 0
# total = 0

# for batch in tqdm(dataloader, desc="Evaluating"):
#     inputs = batch["pixel_values"].to(device)
#     labels = batch["label"].to(device)

#     with torch.no_grad():
#         outputs = model(pixel_values=inputs)
#         preds = outputs.logits.argmax(dim=-1)

#     correct += (preds == labels).sum().item()
#     total += labels.size(0)

# accuracy = 100 * correct / total
# print(f"Accuracy on {total} samples: {accuracy:.2f}%")


# from datasets import load_dataset
# from transformers import AutoImageProcessor, ViTForImageClassification
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from torch.utils.data import DataLoader
# from PIL import Image
# import torch
# from tqdm import tqdm

# # Load ViT model + processor
# model_name = "vit-base-patch16-imagenette"
# model = ViTForImageClassification.from_pretrained(model_name)
# processor = AutoImageProcessor.from_pretrained(model_name,use_fast=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device).eval()

# # Load ImageNet-1k dataset from HuggingFace
# #dataset = load_dataset("imagenet-1k", split="validation[:5%]",trust_remote_code=True)  # Use 5% for speed/testing
# dataset = load_dataset("frgfm/imagenette", '160px', split="validation")

# # Preprocessing (matches ViT expectations)
# # def transform(example):
# #     image = example["image"].convert("RGB")
# #     inputs = processor(image, return_tensors="pt")

# #     #print("DEBUG >>> processor output:", type(inputs["pixel_values"]), inputs["pixel_values"].shape)

# #     inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
# #     #print("AFTER squeeze >>>", type(inputs["pixel_values"]), inputs["pixel_values"].shape)

# #     # Inspect the raw tensor content to make sure it's not a list of tensors
# #     #print("Tensor value (first channel):", inputs["pixel_values"][0])

# #     inputs["label"] = torch.tensor(example["label"])
# #     return inputs
# imagenette_label_names = [
#     "tench", "English springer", "cassette player", "chain saw", "church",
#     "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
# ]

# imagenet_label_map = model.config.label2id 
# imagenette_to_imagenet = {
#     0: 0,
#     1: 107,
#     2: 423,
#     3: 469,
#     4: 555,
#     5: 566,
#     6: 569,
#     7: 571,
#     8: 574,
#     9: 701
# }
#  # maps label names → 0–999 indices

# def transform(example):
#     image = example["image"].convert("RGB")
#     inputs = processor(image, return_tensors="pt")
#     inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)

#     imagenette_label = example["label"]  # 0–9
#     imagenet_label = imagenette_to_imagenet[imagenette_label]
#     inputs["label"] = torch.tensor(imagenet_label)
#     return inputs

# # Apply transform
# #dataset = dataset.map(transform)
# dataset = dataset.map(transform, remove_columns=dataset.column_names)
# dataset.set_format(type="torch")

# # Wrap into DataLoader
# def collate_fn(batch):
#     for i, x in enumerate(batch):
#         print(f"[{i}] pixel_values type: {type(x['pixel_values'])}, shape: {getattr(x['pixel_values'], 'shape', 'N/A')}")
#         if isinstance(x["pixel_values"], list):
#             print(">>> LIST DETECTED <<< type:", type(x["pixel_values"]))
#         else:
#             print(">>> TENSOR SHAPE <<<", x["pixel_values"].shape)
#     pixel_values = torch.stack([x["pixel_values"] for x in batch])
#     labels = torch.stack([x["label"] for x in batch])
#     return pixel_values, labels

# dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

# # Run inference
# correct = 0
# total = 0

# for images, labels in tqdm(dataloader):
#     images, labels = images.to(device), labels.to(device)
#     with torch.no_grad():
#         logits = model(pixel_values=images).logits
#         preds = logits.argmax(dim=-1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

# print(f"Accuracy on ImageNet subset: {100 * correct / total:.2f}%")

# import torch
# from datasets import load_dataset
# from transformers import AutoImageProcessor, ViTForImageClassification, ViTConfig
# from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# from tqdm import tqdm

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Step 1: Load model config and initialize ViT PyTorch model
# model_name = "google/vit-base-patch16-224"
# config = ViTConfig.from_pretrained(model_name)
# model = ViTForImageClassification(config)

# # Step 2: Load Flax weights into PyTorch model
# #load_flax_checkpoint_in_pytorch_model(model, model_name)
# model.to(device)
# model.eval()

# # Step 3: Load processor (use base ViT processor)
# processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# # Step 4: Load Imagenette validation set
# dataset = load_dataset("imagenette", split="validation")

# # Step 5: Define image transform
# transform = Compose([
#     Resize(256),
#     CenterCrop(224),
#     ToTensor(),
#     Normalize(mean=processor.image_mean, std=processor.image_std)
# ])

# # Step 6: Evaluation loop
# correct = 0
# total = 0

# for sample in tqdm(dataset, desc="Evaluating"):
#     image = sample["image"]
#     label = sample["label"]

#     image_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         predicted = outputs.logits.argmax(dim=1).item()

#     correct += int(predicted == label)
#     total += 1

# accuracy = correct / total
# print(f"\n✅ Inference Accuracy on Imagenette Validation Set: {accuracy:.4f}")

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from PIL import Image

# Configuration
model_name = "WinKawaks/vit-tiny-patch16-224"

split = "validation[:200]"  # small subset for quick test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model & processor
model = ViTForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)
model.to(device)
model.eval()

# Load a small subset of Imagenette
dataset = load_dataset("johnowhitaker/imagenette2-320", split="train[:200]")

# Define transform (match ViT preprocessing)
image_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Run inference
correct = 0
total = 0

for sample in tqdm(dataset):
    image = sample["image"]
    label = sample["label"]

    image_tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted = torch.argmax(outputs.logits, dim=1).item()

    if predicted == label:
        correct += 1
    total += 1

accuracy = correct / total
print(f"\n✅ Accuracy on {total} samples: {accuracy:.2%}")
