# import numpy as np

# w = np.fromfile("weights/attn_out_proj.bin", dtype=np.float32)
# print("Loaded shape:", w.shape)          # Should be 200×192 = 38400
# print("Expected total:", 200 * 192)   
from datasets import load_dataset

dataset = load_dataset("zh-plus/tiny-imagenet", split="train", keep_in_memory=True)
dataset = dataset.shuffle(seed=42)
label_info = dataset.features["label"]

print(label_info)                   # Will print ClassLabel object
print(label_info.names[:10])       # Human-readable class names
print(dataset["label"][:10])   

with open("input/labels.txt") as f:
    labels = [int(x.strip()) for x in f.readlines()]
print("First 10 labels in labels.txt:", labels[:10])