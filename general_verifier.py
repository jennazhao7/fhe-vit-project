import numpy as np
x = np.fromfile("input/image_0.bin", dtype=np.float32)
print(x.shape)           # Should be (3, 224, 224) flattened → 150528
print(x.min(), x.max())  # Should be ~ -2 to 2 range if normalized correctly
