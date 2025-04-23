# fhe_vit_runner.py

import torch
from transformers import AutoImageProcessor
from datasets import load_dataset
from PIL import Image
from segmented_vit import SegmentedViTForFHE
import matplotlib.pyplot as plt

# Step 1: Load model and processor
model = SegmentedViTForFHE()
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model.eval()

# Step 2: Load a sample image from Imagenette
dataset = load_dataset("johnowhitaker/imagenette2-320", split="train")  # consistent source
image_data = dataset[0]
image = image_data["image"]  # PIL.Image object
image.save("sample.jpg")

# Optional: Display the image
image.show()

# Step 3: Preprocess the image
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]

# Step 4: Dummy FHE preprocessing
def fhe_dummy_encrypt_and_scale(x):
    print("Simulating dummy encryption on tensor shape:", x.shape)
    return x * 1.0  # Placeholder for encrypted tensor (still torch.Tensor)

pixel_values[0] = fhe_dummy_encrypt_and_scale(pixel_values[0])

# Step 5: Run inference through segmented ViT with mock FHE layers
with torch.no_grad():
    outputs = model(pixel_values)
    predicted_class = torch.argmax(outputs, dim=1)

# Step 6: Display result
print("Predicted class:", predicted_class.item())
plt.imshow(image)
plt.title(f"Predicted Class: {predicted_class.item()}")
plt.axis("off")
plt.show()