
# FHE ViT Inference: Privacy-Preserving Vision Transformer

This project performs inference on a Vision Transformer (ViT) using Fully Homomorphic Encryption (FHE) via OpenFHE. The model processes a single image sample end-to-end while preserving data privacy during computation.

---

## Project Overview

- **Model**: Vision Transformer
- **Novelty**: Fully Homomorphic Encryption using OpenFHE
- **Input**: Preprocessed image vector (e.g., `image_2.bin`)
- **Output**: Predicted label & intermediate values written to `full_inference_output.txt`

---

## Requirements

### System
- Ubuntu 20.04+ or WSL (Linux)
- C++17 compiler (e.g., `g++`, `clang++`)
- CMake ≥ 3.10
- Git

### C++ Libraries
- OpenFHE (installed from prebuilt release — no git clone required)

---
## Instructions for Running test_example
- Clone the repo with submodules

```bash 
git clone --recurse-submodules https://github.com/jennazhao7/fhe-vit-project.git 
cd fhe-vit-project
```

- Build OpenFHE

```bash 
cd openfhe-development
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_DEMOS=OFF -DBUILD_TESTING=OFF -DHAVE_STD_REGEX=ON -DRUN_HAVE_STD_REGEX=1
make -j2
cd ../../
```
- Build the project

```bash 
mkdir -p build && cd build
cmake ..
make -j2
```
- Run the test app

```bash 
./test_example
```

---
## Installing OpenFHE (No Git Needed)

1. **Download prebuilt OpenFHE:**

Go to: [https://github.com/openfheorg/openfhe-development/releases](https://github.com/openfheorg/openfhe-development/releases)

Choose: openfhe-ubuntu-22.04.tar.gz


2. **Extract and move to install location:**


tar -xvzf openfhe-ubuntu-22.04.tar.gz
mv openfhe-ubuntu-22.04 $HOME/openfhe-install

## Installing this fhe-vit project
Clone from git path: 
- git clone https://github.com/jennazhao7/fhe-vit-project.git
- cd fhe-vit-project

Create and enter build folder: 
- mkdir build && cd build

Run CMake with OpenFHE path: 
- cmake .. -DOpenFHE_DIR=$HOME/openfhe-install/lib/cmake/OpenFHE

Compile the project: 
- make -j

Once built, run the executable: 
- ./fhe_inference

Output results to:
- full_inference_output.txt
