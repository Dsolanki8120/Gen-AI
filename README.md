Image Caption Generation using Vision Transformer (ViT) and Transformer Decoder

A multimodal Deep Learning project that generates natural language descriptions for images by bridging Computer Vision and Natural Language Processing (NLP). This project replaces traditional CNN-LSTM architectures with a fully attention-based mechanism.

* Project Overview:
*   **Goal:Generate accurate and fluent captions for input images.
*   **Architecture:** Hybrid Transformer Model.
    *   **Encoder:** Vision Transformer (ViT-Base) pre-trained on ImageNet.
    *   **Decoder:** Custom Transformer Decoder trained from scratch.
*   **Dataset:** Flickr8k.
*   **Status:** Phase 1 Complete (Exceeded BLEU-1 Target).

## Results
We evaluated the model on the Flickr8k test set (200 unseen images) using standard BLEU metrics.

| Metric | Score Achieved | Interpretation |
| :--- | :--- | :--- |
| **BLEU-1** | **64.90** | High accuracy in object detection. |
| **BLEU-2** | **51.94** | Strong phrase matching. |
| **BLEU-4** | **33.00** | Fluent sentence generation. |

## 🏗️ Model Architecture
The system follows an Encoder-Decoder pipeline:
1.  **Image Preprocessing:** Images are resized to `224x224` and patched into `16x16` grids.
2.  **ViT Encoder:** Extracts global visual features (Dimension: 768).
3.  **Projection Layer:** Maps visual features to the decoder dimension (768 → 512).
4.  **Transformer Decoder:** Generates text word-by-word using Self-Attention (for text context) and Cross-Attention (for visual context).

---

## Getting Started

### Prerequisites
*   Python 3.8+
*   PyTorch (CUDA supported recommended)

### Installation
Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/Dsolanki8120/Gen-AI.git
   cd Gen-AI
   pip install torch torchvision transformers pandas pillow nltk tqdm


   Dataset Setup
   Due to size constraints, the Flickr8k dataset is not included in this repo.
   Dataset Link: Kaggle - https://www.kaggle.com/datasets/adityajn105/flickr8k
   
   Training the Model
      To train the model from scratch (10 Epochs):
         python train.py
   Inference (Testing on Random Images)
      To generate a caption for a random image from the dataset:
         python inference.py
   Evaluation (BLEU Scores)
      To calculate BLEU-1 to BLEU-4 scores on the test set:
         python evaluate.py

---------------------------------------------------------------------------------------------------------------------

   Project Structure
         model.py: Defines the ImageCaptionModel, ViT Encoder, and Transformer Decoder.
         dataset.py: Custom PyTorch Dataset class for loading Flickr8k images and tokenizing captions.
         train.py: Main training loop using Teacher Forcing and Cross-Entropy Loss.
         inference.py: Script to generate captions for new images.
         evaluate.py: Script to calculate BLEU scores using NLTK.

----------------------------------------------------------------------------------------------------------------------
Team Members:
      Manish Kumar
      Suradkar Sharvil Sanjay
      Anshul Jose
      Shivam Mandloi
      Deepak Solanki








