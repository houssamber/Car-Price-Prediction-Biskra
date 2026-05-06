import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import json

# ═══════════════════════════════════════════════════════
#  حساب معاملات التطبيع من بيانات التدريب
#  هذه القيم ضرورية لأن النموذج يتعلم على أسعار مطبّعة
#  ونحتاجها لاحقاً في app.py لعكس التطبيع وإعطاء السعر الحقيقي
# ═══════════════════════════════════════════════════════
_df_check = pd.read_csv('combined_data.csv')
PRICE_MEAN = float(_df_check['price'].mean())   # ≈ 197.93
PRICE_STD  = float(_df_check['price'].std())    # ≈ 123.82

# نحفظ هذه القيم في ملف صغير حتى يقرأها app.py
with open('price_scaler.json', 'w') as f:
    json.dump({'mean': PRICE_MEAN, 'std': PRICE_STD}, f)

print(f" معاملات التطبيع: mean={PRICE_MEAN:.2f}, std={PRICE_STD:.2f}")

# ═══════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════
class CarDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir   = root_dir
        self.transform  = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            return self.__getitem__((idx + 1) % len(self.data_frame))

        # ✅ تطبيع السعر: (price - mean) / std
        # بدلاً من إدخال 220 أو 190 مباشرة، ندخل قيمة قريبة من الصفر
        # هذا يجعل تدريب النموذج أسرع وأكثر استقراراً
        raw_price = float(self.data_frame.iloc[idx, 1])
        normalized_price = (raw_price - PRICE_MEAN) / PRICE_STD
        price = torch.tensor(normalized_price, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, price


# ═══════════════════════════════════════════════════════
#  Data Augmentation
# ═══════════════════════════════════════════════════════
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ═══════════════════════════════════════════════════════
#  DataLoader
# ═══════════════════════════════════════════════════════
dataset      = CarDataset('combined_data.csv', '.', data_transforms)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)

# ═══════════════════════════════════════════════════════
#  النموذج: ResNet18 مع Transfer Learning
# ═══════════════════════════════════════════════════════
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)   # مخرج واحد = السعر المطبَّع

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model  = model.to(device)
print(f"🖥️  جهاز التدريب: {device}")

# ═══════════════════════════════════════════════════════
#  الخسارة والمحسِّن
# ═══════════════════════════════════════════════════════
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# ═══════════════════════════════════════════════════════
#  حلقة التدريب — 50 Epoch
# ═══════════════════════════════════════════════════════
num_epochs = 50
print(f"\n🚀 بدء التدريب على {len(dataset)} صورة ...\n")

loss_history = []   # نحفظ الخسارة لرسم المنحنى لاحقاً

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, prices in train_loader:
        images = images.to(device)
        prices = prices.to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, prices)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        # نحوّل الخسارة المطبّعة إلى مليون سنتيم لفهمها
        approx_error_millions = (avg_loss ** 0.5) * PRICE_STD
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Loss: {avg_loss:.4f} | "
              f"خطأ تقريبي: ±{approx_error_millions:.1f} مليون سنتيم")

# ═══════════════════════════════════════════════════════
#  حفظ النموذج وسجل الخسارة
# ═══════════════════════════════════════════════════════
torch.save(model.state_dict(), 'car_price_model.pth')

with open('loss_history.json', 'w') as f:
    json.dump(loss_history, f)

print("\n" + "="*50)
print(" تم حفظ النموذج: car_price_model.pth")
print(" تم حفظ معاملات التطبيع: price_scaler.json")
print(" تم حفظ سجل الخسارة: loss_history.json")
print("="*50)
print("\n▶ شغّل الآن: streamlit run app.py")
