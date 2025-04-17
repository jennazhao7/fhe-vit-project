# fhe_vit_runner.py

import torch
import numpy as np
from PIL import Image
from transformers import ViTModel, ViTImageProcessor
from openfhe import *  # Assuming you have OpenFHE Python bindings installed

# Step 1: Load processor and model
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)

# Step 2: Load image
image = Image.open("sample_image.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]

# Step 3: Optional encryption of pixel_values
def fhe_dummy_encrypt_and_scale(x):
    params = CCParamsBFVRNS()
    params.SetPlaintextModulus(65537)
    params.SetMultiplicativeDepth(2)
    cc = GenCryptoContext(params)
    cc.Enable(ENCRYPTION)
    cc.Enable(SHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    flat = x.flatten().tolist()
    pt = cc.MakePackedPlaintext([int(v * 255) % 65537 for v in flat])
    ct = cc.Encrypt(keys.publicKey, pt)

    ct2 = cc.EvalMult(ct, 2)
    pt2 = cc.Decrypt(keys.secretKey, ct2)
    pt2.SetLength(len(flat))
    arr = np.array(pt2.GetPackedValue()) / 255.0
    return torch.tensor(arr).view_as(x)

# Example: Encrypt and scale only the first image
with torch.no_grad():
    pixel_values[0] = fhe_dummy_encrypt_and_scale(pixel_values[0])

# Step 4: Run forward pass manually through parts of the model
model.eval()
with torch.no_grad():
    # Pass through encoder directly (not classification head)
    outputs = model(pixel_values=pixel_values)
    last_hidden_state = outputs.last_hidden_state

print("Final ViT embedding shape:", last_hidden_state.shape)

from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained(model_name)

with torch.no_grad():
    pixel_values[0] = fhe_dummy_encrypt_and_scale(pixel_values[0])
    logits = model(pixel_values=pixel_values).logits
    predicted_class = torch.argmax(logits, dim=1)

print("Predicted class:", predicted_class)