import torch
import random
import os
from PIL import Image
from dataset import FlickrDataset
from model import ImageCaptionModel
from transformers import ViTImageProcessor

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset (just to get vocabulary)
# root_dir must match your folder name "Images"
dataset = FlickrDataset(root_dir="Images", caption_file="captions.txt")
vocab = dataset.vocab
vocab_size = len(vocab.stoi)

# Load Model
model = ImageCaptionModel(embed_size=512, hidden_size=512, vocab_size=vocab_size, num_layers=6, num_heads=8).to(device)

if os.path.exists("caption_model.pth"):
    model.load_state_dict(torch.load("caption_model.pth", map_location=device))
    print("Model loaded successfully.")
else:
    print("Error: caption_model.pth not found. Train the model first!")
    exit()

model.eval()
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def generate_caption(image_path, max_length=20):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Error opening image: {e}"

    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)
    
    # Encode
    with torch.no_grad():
        vit_output = model.vit(pixel_values=pixel_values).last_hidden_state
        image_features = model.projection(vit_output)
    
    # Decode
    start_token = vocab.stoi["<start>"]
    generated_caption = [start_token]
    
    for _ in range(max_length):
        inputs = torch.tensor(generated_caption).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embeddings = model.embed(inputs) + model.positional_enc[:, :inputs.size(1), :]
            outputs = model.decoder(tgt=embeddings, memory=image_features)
            predictions = model.linear(outputs[:, -1, :])
            predicted_id = predictions.argmax(1).item()
        
        if vocab.itos[predicted_id] == "<end>":
            break
            
        generated_caption.append(predicted_id)
            
    return " ".join([vocab.itos[idx] for idx in generated_caption if idx not in [0, 1, 2, 3]])

# --- Test on a Random Image ---
if __name__ == "__main__":
    # Pick a random image from the folder
    all_images = os.listdir("Images")
    random_image = random.choice(all_images)
    img_path = os.path.join("Images", random_image)
    
    print(f"\nTesting on image: {random_image}")
    caption = generate_caption(img_path)
    print(f"Generated Caption: {caption}")
    
    # Show the image (optional, works if you have a display)
    # Image.open(img_path).show()