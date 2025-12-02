import torch
import torch.nn as nn
from transformers import ViTModel

class ImageCaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, num_heads):
        super(ImageCaptionModel, self).__init__()
        
        # 1. ENCODER (ViT) - Page 3
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # Freeze ViT initially as per "Training Strategy"
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # 2. PROJECTION LAYER - Page 3
        # ViT output dim is 768. Map to embed_size (decoder input)
        self.projection = nn.Linear(768, embed_size)
        
        # 3. DECODER - Page 3
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Positional encoding is required for Transformer Decoders
        self.positional_enc = nn.Parameter(torch.zeros(1, 100, embed_size)) # Max len 100
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.linear = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, captions):
        # --- Encoder Pass ---
        # ViT output: last_hidden_state is (Batch, 197, 768)
        # 197 = 196 patches + 1 CLS token
        vit_output = self.vit(pixel_values=images).last_hidden_state
        
        # Project visual features to decoder dimension
        # (Batch, 197, 768) -> (Batch, 197, embed_size)
        image_features = self.projection(vit_output) 
        
        # --- Decoder Pass ---
        # Create Embeddings for captions + Positional Encodings
        # (Batch, Seq_Len) -> (Batch, Seq_Len, Embed_Size)
        embeddings = self.dropout(self.embed(captions) + self.positional_enc[:, :captions.size(1), :])
        
        # Masking: Prevent looking at future tokens
        tgt_mask = self.make_mask(captions).to(images.device)
        
        # Decode
        # tgt = Text Embeddings, memory = Image Features
        output = self.decoder(tgt=embeddings, memory=image_features, tgt_mask=tgt_mask)
        
        # Prediction
        outputs = self.linear(output)
        return outputs

    def make_mask(self, sz):
        # Standard causal mask for transformer
        seq_len = sz.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask