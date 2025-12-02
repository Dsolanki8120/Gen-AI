import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import ViTImageProcessor
import os
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<start>", 2: "<end>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<start>": 1, "<end>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in sentence:
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = text.split()
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

class FlickrDataset(Dataset):
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        
        # --- NEW CODE: Filter out missing images ---
        print("Checking dataset integrity...")
        # Check if files exist
        self.df['file_exists'] = self.df['image'].apply(lambda x: os.path.exists(os.path.join(root_dir, x)))
        
        # Count missing
        missing_count = len(self.df) - self.df['file_exists'].sum()
        if missing_count > 0:
            print(f"Warning: Removed {missing_count} captions because images were missing.")
        
        # Keep only existing ones
        self.df = self.df[self.df['file_exists']].reset_index(drop=True)
        # -------------------------------------------

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        # Initialize Vocabulary
        self.vocab = Vocabulary(freq_threshold)
        tok_captions = [str(cap).lower().replace('.', '').split() for cap in self.captions]
        self.vocab.build_vocabulary(tok_captions)
        
        self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img_path = os.path.join(self.root_dir, img_id)
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupted images (rare case)
            print(f"Error loading image {img_id}: {e}")
            # Return a dummy tensor or handle as needed, but usually filtering init handles it.
            # Here we'll just crash with a clearer message if filtering failed
            raise e

        pixel_values = self.feature_extractor(images=img, return_tensors="pt").pixel_values.squeeze(0)

        caption = str(caption).lower().replace('.', '')
        caption_vec = [self.vocab.stoi["<start>"]] 
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<end>"]]

        return pixel_values, torch.tensor(caption_vec)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Filter out None values in case of loading errors
        batch = [item for item in batch if item is not None]
        
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets