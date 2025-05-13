import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
from collections import Counter
 

def get_transforms(processor):
    return Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)
        outputs = model(pixel_values=inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            outputs = model(pixel_values=inputs)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if total < 32:  # print only for the first few batches
                print("Preds:", preds.tolist())
                print("Labels:", labels.tolist())

    return 100 * correct / total


def main():
    # Load Tiny ImageNet dataset
    dataset = load_dataset("zh-plus/tiny-imagenet", split="train", keep_in_memory=True)
    dataset = dataset.shuffle(seed=42)
    # Deterministic 70/30 split
    train_size = int(0.7 * len(dataset))
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, len(dataset)))

    # Model + processor
    model_name = "WinKawaks/vit-tiny-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)

    transform = get_transforms(processor)
    print(train_data.features["label"])  # Shows the label type (e.g., ClassLabel)
    print(Counter(train_data["label"]))
    # Set dataset transforms for batching
    train_data.set_transform(lambda examples: {
        "pixel_values": [transform(img.convert("RGB")) for img in examples["image"]],
        "label": examples["label"]
    })
    val_data.set_transform(lambda examples: {
        "pixel_values": [transform(img.convert("RGB")) for img in examples["image"]],
        "label": examples["label"]
    })

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=200,  # Tiny ImageNet has 200 classes
        ignore_mismatched_sizes=True
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    max_epochs = 50            # Safety cap
    min_loss = 0.2             # Stop if loss falls below this
    patience = 3               # Stop if no improvement in this many epochs
    best_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}")
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        # Check for improvement
        if train_loss < best_loss - 1e-4:  # Significant improvement
            best_loss = train_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Stopping conditions
        if train_loss < min_loss:
            print(f"Stopping: loss {train_loss:.4f} < threshold {min_loss}")
            break
        if no_improve_epochs >= patience:
            print(f"Stopping: no improvement for {patience} epochs")
            break
    torch.save(model.state_dict(), "best_model.pt")

if __name__ == "__main__":
    main()