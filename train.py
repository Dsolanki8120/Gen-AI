import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import FlickrDataset, MyCollate
from model import ImageCaptionModel
from tqdm import tqdm

# Hyperparameters
embed_size = 512
hidden_size = 512
num_heads = 8
num_layers = 6 # As per Page 3
learning_rate = 3e-4
num_epochs = 10
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
# root_dir must match your folder name "Images" exactly (Case Sensitive)
dataset = FlickrDataset(root_dir="Images", caption_file="captions.txt")
pad_idx = dataset.vocab.stoi["<PAD>"]
vocab_size = len(dataset.vocab.stoi)

# Split (6000 train, 1000 val, 1000 test - Page 2)
train_size = 6000
val_size = 1000
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=MyCollate(pad_idx=pad_idx), shuffle=True)

# Initialize Model
model = ImageCaptionModel(embed_size, hidden_size, vocab_size, num_layers, num_heads).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    model.train()
    
    # Optional: Unfreeze ViT layers later (Page 4, Step 8)
    if epoch == 5:
        print("Unfreezing ViT last layers for fine-tuning...")
        for param in model.vit.parameters():
            param.requires_grad = True

    for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader)):
        imgs = imgs.to(device)
        captions = captions.to(device)

        # Teacher Forcing Strategy (Page 4, Step 6)
        # Input to decoder: <start> A cat is...
        # Target: A cat is ... <end>
        outputs = model(imgs, captions[:, :-1]) 
        targets = captions[:, 1:]

        optimizer.zero_grad()
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")

# Save Model
torch.save(model.state_dict(), "caption_model.pth")