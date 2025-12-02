import torch
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from transformers import ViTImageProcessor
from dataset import FlickrDataset # To get vocabulary
from model import ImageCaptionModel

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dir = "Images"
caption_file = "captions.txt"

# 1. Load Vocabulary & Model
print("Loading Model...")
temp_dataset = FlickrDataset(root_dir=image_dir, caption_file=caption_file)
vocab = temp_dataset.vocab
vocab_size = len(vocab.stoi)

model = ImageCaptionModel(embed_size=512, hidden_size=512, vocab_size=vocab_size, num_layers=6, num_heads=8).to(device)
model.load_state_dict(torch.load("caption_model.pth", map_location=device))
model.eval()

feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# 2. Group Captions by Image (The "1-vs-5" Logic)
print("Grouping captions...")
df = pd.read_csv(caption_file)
# Create a dictionary: image_filename -> [caption1, caption2, caption3, caption4, caption5]
image_to_captions = {}
for idx, row in df.iterrows():
    img_name = row['image']
    caption = str(row['caption']).lower().replace('.', '').split()
    
    if img_name not in image_to_captions:
        image_to_captions[img_name] = []
    image_to_captions[img_name].append(caption)

# 3. Filter only existing images & Select Test Set
all_images = list(image_to_captions.keys())
existing_images = [img for img in all_images if os.path.exists(os.path.join(image_dir, img))]

# Use last 200 images for testing (simulating a test split)
test_images = existing_images[-200:] 

print(f"Evaluating on {len(test_images)} images using 5 references each...")

references = []
hypotheses = []

# 4. Inference Loop
for img_name in tqdm(test_images):
    img_path = os.path.join(image_dir, img_name)
    
    # Process Image
    img = Image.open(img_path).convert("RGB")
    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        vit_output = model.vit(pixel_values=pixel_values).last_hidden_state
        image_features = model.projection(vit_output)
        
        # Greedy Decode
        start_token = vocab.stoi["<start>"]
        generated_ids = [start_token]
        
        for _ in range(20):
            inputs = torch.tensor(generated_ids).unsqueeze(0).to(device)
            embeddings = model.embed(inputs) + model.positional_enc[:, :inputs.size(1), :]
            outputs = model.decoder(tgt=embeddings, memory=image_features)
            predicted_id = model.linear(outputs[:, -1, :]).argmax(1).item()
            
            if vocab.itos[predicted_id] == "<end>":
                break
            generated_ids.append(predicted_id)
            
    # Convert IDs to Words
    pred_caption = [vocab.itos[idx] for idx in generated_ids if idx not in [0, 1, 2, 3]]
    
    # Store Hypothesis
    hypotheses.append(pred_caption)
    
    # Store References (List of lists)
    # The references are already tokenized in step 2
    references.append(image_to_captions[img_name])

# 5. Calculate BLEU
bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

print(f"\nFinal Results:")
print(f"BLEU-1: {bleu1*100:.2f}")
print(f"BLEU-2: {bleu2*100:.2f}")
print(f"BLEU-4: {bleu4*100:.2f}")