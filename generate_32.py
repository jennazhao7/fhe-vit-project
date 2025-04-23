import numpy as np

# Create a 32x32 matrix with small random values (e.g., Gaussian or uniform)
out_dim = 32
in_dim = 32

# You can use either normal or uniform random initialization
weights = np.random.uniform(low=-0.1, high=0.1, size=(out_dim, in_dim)).astype(np.float32)

# Save as binary file (row-major)
weights.tofile("weights/patch_embed_32x32.bin")

print("Generated weights/patch_embed_32x32.bin with shape", weights.shape)