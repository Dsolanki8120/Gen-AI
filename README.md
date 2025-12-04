Image Caption Generation using Vision Transformer (ViT) and Transformer Decoder

A multimodal Deep Learning project that generates natural language descriptions for images by bridging Computer Vision and Natural Language Processing (NLP). This project replaces traditional CNN-LSTM architectures with a fully attention-based mechanism.

 Project Overview:
    Goal: Generate accurate and fluent captions for input images.
   Architecture: Hybrid Transformer Model.
        Encoder: Vision Transformer (ViT-Base) pre-trained on ImageNet.
        Decoder: Custom Transformer Decoder trained from scratch.
        Dataset: Flickr8k.
        Status: Phase 1 Complete (Exceeded BLEU-1 Target).
Results:
We evaluated the model on the Flickr8k test set (200 unseen images) using standard BLEU metrics.

| Metric |Score Achieved** | Interpretation |
| :--- | :--- | :--- |
|BLEU-1 |64.90 | High accuracy in object detection. |
|BLEU-2| 51.94| Strong phrase matching. |
|BLEU-4 | 33.00 | Fluent sentence generation. |


Model Architecture :
The system follows an Encoder-Decoder pipeline:
1.  Image Preprocessing: Images are resized to `224x224` and patched into `16x16` grids.
2.  ViT Encoder: Extracts global visual features (Dimension: 768).
3.  Projection Layer: Maps visual features to the decoder dimension (768 → 512).
4.  Transformer Decoder: Generates text word-by-word using Self-Attention (for text context) and Cross-Attention (for visual context).


Prerequisites :
  Python 3.8+
  PyTorch (CUDA supported recommended)

Installation :
Clone the repository and install dependencies:

  git clone https://github.com/Dsolanki8120/Gen-AI.git
  cd Gen-AI
  pip install torch torchvision transformers pandas pillow nltk tqdm 



Dataset Setup
  Due to size constraints, the Flickr8k dataset is not included in this repo.
  Data set: https://www.kaggle.com/datasets/adityajn105/flickr8k  

Training the Model:
  To train the model from scratch (10 Epochs): 
    python train.py

Inference (Testing on Random Images):
  To generate a caption for a random image from the dataset:
     python inference.py 

Evaluation (BLEU Scores):
  To calculate BLEU-1 to BLEU-4 scores on the test set:
    python evaluate.py 

Project Structure:
      model.py: Defines the ImageCaptionModel, ViT Encoder, and Transformer Decoder.
      dataset.py: Custom PyTorch Dataset class for loading Flickr8k images and tokenizing captions.
      train.py: Main training loop using Teacher Forcing and Cross-Entropy Loss.
      inference.py: Script to generate captions for new images.
      evaluate.py: Script to calculate BLEU scores using NLTK.  


Comparison with State-of-the-Art: 

    To benchmark our model, we compared its performance against BLIP-2 (Salesforce Research, 2023), a leading massive multimodal model.
    Our Model: Trained on Flickr8k (8,000 images). Focuses on factual, object-centric descriptions.
Qualitative Results: 

  | Image | Our Model Prediction (Descriptive) | SOTA: BLIP-2 Prediction | Analysis |
  | img ="children_hay.jpg" |"Three happy children are playing with dry grass in an open field, throwing it in the air and enjoying the sunny day."| "Children joyfully toss hay into the air while playing outdoors, surrounded by bright sunlight and an open sky." | Our model captures the count (three) and action correctly. BLIP-2 adds atmospheric details like "bright sunlight." |
  | img="pig_crowd.jpg"     |"A group of children are trying to catch a small pig while many people stand behind a fence watching them." | "Children chase and grab a small pig in a fenced outdoor area as a large crowd gathers around to watch the playful event." | Our model correctly identifies the "catch" action. BLIP-2 infers the context of a "playful event." |
  | img ="skater.jpg"       | "A person is doing a bicycle stunt by riding high on a wall in an outdoor area." | "A cyclist performs an impressive wall-ride trick on a BMX bike, climbing high up a brick wall in a sunny outdoor setting."| Our model identifies "bicycle stunt." BLIP-2 recognizes the specific bike type ("BMX") and trick name ("wall-ride"). |
  | img ="skiing.jpg"       | "Two people wearing ski gear are sitting on a ski lift, hanging in the air with their skis on." | "Two skiers ride a high ski lift, dressed in helmets and goggles, with their skis dangling against the clear blue sky." | Our model uses simple words ("hanging in the air"). BLIP-2 uses more descriptive vocabulary ("dangling against clear blue sky"). |
  | img="dog_shake.jpg"     | "A dog is shaking its body in the water, making splashes all around." | "A wet dog vigorously shakes off water, sending droplets flying in all directions while standing in a shallow pond."| Success Case: Our model captures the exact action. BLIP-2 enhances it with adverbs like "vigorously." |

 Conclusion on Comparison
    While BLIP-2 produces more "poetic" and detailed captions due to its massive training data, our ViT + Transformer model successfully captures the core semantic meaning (Who, What, Where) of every image. This demonstrates that for assistive applications (like explaining a scene to a visually impaired person), our model provides the essential information accurately and efficiently.

Team Members:

      Manish Kumar
      Suradkar Sharvil Sanjay
      Anshul Jose
      Deepak Solanki
      Shivam Mandloi
Future Work
    Implement Beam Search for better caption quality.
    Train on larger datasets like MS-COCO.
    Optimize using Reinforcement Learning (CIDEr optimization).





    
