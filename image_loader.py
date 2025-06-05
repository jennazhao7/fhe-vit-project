# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# from datasets import load_dataset
# from transformers import ViTImageProcessor
# import numpy as np

# processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224")
# transform = Compose([
#     Resize(256),
#     CenterCrop(224),
#     ToTensor(),
#     Normalize(mean=processor.image_mean, std=processor.image_std)
# ])

# dataset = load_dataset("zh-plus/tiny-imagenet", split="train").shuffle(seed=42)
# dataset = dataset.select(range(200))  # pick N=20 samples for now

# with open("input/labels.txt", "w") as label_file:
#     for i, example in enumerate(dataset):
#         img = example["image"].convert("RGB")
#         arr = transform(img).numpy().astype(np.float32)
#         print(arr.shape)
#         arr.tofile(f"input/image_{i}.bin")
#         label_file.write(f"{example['label']}\n")

import torch
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification

# === Load pretrained model and processor ===
model = ViTForImageClassification.from_pretrained(
    "WinKawaks/vit-tiny-patch16-224", num_labels=200, ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224")

transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std)
])

dataset = load_dataset("zh-plus/tiny-imagenet", split="train").shuffle(seed=42)
# dataset = dataset.select(range(200))  # pick N=20 samples for now

with open("input/labels.txt", "w") as label_file:
    for i, example in enumerate(dataset):
        img = transform(example["image"].convert("RGB")).unsqueeze(0)  # shape: [1, 3, 224, 224]
        # with torch.no_grad():
        #     patch_embed = model.vit.embeddings.patch_embeddings(img)  # [1, 196, 192]
        #     patch_embed = patch_embed[:, 0, :]  # CLS token is added later, use first token embedding
        #     vec = patch_embed.squeeze(0).numpy().astype(np.float32)  # [192]

        # print(f"[INFO] image_{i}.bin shape: {vec.shape}")
        # vec.tofile(f"input/image_{i}.bin")
        label_file.write(f"{example['label']}\n")
        img_np = img.squeeze(0).numpy().astype(np.float32)  # [3, 224, 224]
        img_np.tofile(f"input/image_{i}.bin")