# **Transfer Learning**
### Digit 7-9

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import pandas as pd
import numpy as np

df = pd.read_csv('digit_train.csv')

target_labels = [7, 8, 9]
df_filtered = df[df['label'].isin(target_labels)].reset_index(drop=True)

X_train = torch.tensor(df_filtered.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train = torch.tensor(df_filtered.iloc[:, 0].values, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.alexnet()
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)

model.load_state_dict(torch.load('pretrain.pth', map_location=device))

for param in model.features.parameters():
    param.requires_grad = False

model = model.to(device)

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(X_train.size(0))

    total_loss, correct, total = 0, 0, 0

    for i in range(0, X_train.size(0), BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch_x, batch_y = X_train[indices], y_train[indices]

        batch_x = F.interpolate(batch_x, size=(224, 224), mode='bilinear').repeat(1, 3, 1, 1) / 255.0

        optimizer.zero_grad()
        outputs = model(batch_x.to(device))
        loss = criterion(outputs, batch_y.to(device))
        loss.backward()

        # Mask Apply
        with torch.no_grad():
            model.classifier[6].weight.grad[0:7, :] = 0
            model.classifier[6].bias.grad[0:7] = 0

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y.to(device)).sum().item()

    avg_loss = total_loss / (len(X_train) / BATCH_SIZE)
    acc = 100. * correct / total
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Acc = {acc:.2f}%")

torch.save(model.state_dict(), 'transfer.pth')

# **Testing (TL)**

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

IMAGE_FOLDER = 'TL-Testing'
WEIGHTS_FILE = 'transfer.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.alexnet()
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]

plt.figure(figsize=(15, 10))
num_to_show = min(len(image_files), 10)

for i in range(num_to_show):
    img_name = image_files[i]
    img_path = os.path.join(IMAGE_FOLDER, img_name)

    raw_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)

    plt.subplot(2, 5, i + 1)
    plt.imshow(raw_img)
    plt.title(f"File: {img_name}\nPred: {predicted.item()}")
    plt.axis('off')

plt.tight_layout()
plt.show()