import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from dataset import SeepDataset
from model import UNet

root_dir = os.getcwd()
print("root here",root_dir)
train_image_dir = os.path.join(root_dir, 'train_images_256')
print("train here",train_image_dir)
# train_mask_dir = "/Users/sharvitomar/Desktop/Sem4/cgg/Image_Segmentation/train_masks_256"
# val_image_dir= train_image_dir
# val_mask_dir= train_mask_dir
# test_image_dir= train_image_dir
# test_mask_dir= train_mask_dir


train_mask_dir = os.path.join(root_dir, 'train_masks_256')
val_image_dir = os.path.join(root_dir, 'train_images_256')
val_mask_dir = os.path.join(root_dir, 'train_masks_256')
test_image_dir = os.path.join(root_dir, 'train_images_256')
test_mask_dir = os.path.join(root_dir, 'train_masks_256')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the DataLoader to load the data in batches
train_data = SeepDataset(root_dir)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

val_data = SeepDataset(root_dir)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
model = UNet()
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
print('Training complete')

# Evaluate the model on the test set
test_data = SeepDataset(root_dir)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

model.eval()
test_loss = 0.0
test_corrects = 0

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

with torch.set_grad_enabled(False):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    _, preds = torch.max(outputs, 1)

test_loss += loss.item() * inputs.size(0)
test_corrects += torch.sum(preds == labels.data)
test_loss = test_loss / len(test_loader.dataset)
test_acc = test_corrects.double() / len(test_loader.dataset)

print('Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))
