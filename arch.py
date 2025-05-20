from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224")
#print(model)
for name, param in model.named_parameters():
    print(name, param.shape)