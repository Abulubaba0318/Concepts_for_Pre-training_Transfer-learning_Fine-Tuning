# **Fine-Tuning**
### Roman Digit i-ix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import pandas as pd
import numpy as np

df_roman = pd.read_csv('droman_train.csv')
roman_map = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9}

df_roman = df_roman[df_roman['label'].isin(roman_map.keys())].reset_index(drop=True)
y_train = torch.tensor(df_roman['label'].map(roman_map).values, dtype=torch.long)

X_train_raw = df_roman.iloc[:, 1:].values.astype(np.float32)
if np.mean(X_train_raw) > 127: X_train_raw = 255.0 - X_train_raw
X_train = torch.tensor(X_train_raw).reshape(-1, 1, 28, 28)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.alexnet()
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
model.load_state_dict(torch.load('transfer.pth', map_location=device))

for i, param in enumerate(model.features.parameters()):
    param.requires_grad = (i >= 8)

for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

optimizer = optim.Adam([
    {'params': model.features.parameters(), 'lr': 0.000001}, 
    {'params': model.classifier.parameters(), 'lr': 0.00001}  
])
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    total_loss, correct = 0, 0
    
    for i in range(0, X_train.size(0), 32):
        indices = permutation[i:i + 32]
        batch_x, batch_y = X_train[indices].to(device), y_train[indices].to(device)
        batch_x = F.interpolate(batch_x, size=(224, 224), mode='bilinear').repeat(1, 3, 1, 1) / 255.0
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss/(len(X_train)/32):.4f}, Acc = {100.*correct/len(X_train):.2f}%")

torch.save(model.state_dict(), 'finetuned.pth')

# **Testing (FT)**

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt

IMAGE_FOLDER = 'FT-Testing'
WEIGHTS_FILE = 'finetuned.pth'

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

def predict_image(img_path):
    raw_img = Image.open(img_path).convert('L')
    img_np = np.array(raw_img)

    if np.mean(img_np) > 127:
        processed_img = ImageOps.invert(raw_img)
    else:
        processed_img = raw_img

    processed_img = processed_img.convert('RGB')
    input_tensor = transform(processed_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        prob, predicted = torch.max(probabilities, 1)

    return processed_img, predicted.item(), prob.item()

if not os.path.exists(IMAGE_FOLDER):
    print(f"Error: Folder '{IMAGE_FOLDER}' not exist!")
else:
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    plt.figure(figsize=(15, 12))
    num_to_show = min(len(image_files), 12)

    for i in range(num_to_show):
        img_path = os.path.join(IMAGE_FOLDER, image_files[i])

        final_img, pred, conf = predict_image(img_path)

        plt.subplot(3, 4, i + 1)
        plt.imshow(final_img, cmap='gray')
        plt.title(f"Pred: {pred}\nConf: {conf*100:.2f}%")
        plt.axis('off')

    plt.tight_layout()
    plt.show()