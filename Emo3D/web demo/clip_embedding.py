import torch
import clip
import numpy as np
import os

def get_text_embedding(text):
    print(f"Processing text: {text}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model loaded successfully")
    
    # Tokenize and encode text
    print("Tokenizing text...")
    inputs = clip.tokenize(text).to(device)
    
    with torch.no_grad():
        # Get text features
        print("Generating text features...")
        text_features = model.encode_text(inputs)
        
        # Convert to numpy array
        text_features = text_features.cpu().numpy()
        
        # Get current directory
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'text_embedding.npy')
        
        # Save to file
        print(f"Saving embedding to: {save_path}")
        np.save(save_path, text_features)
        print(f"Embedding saved successfully. Shape: {text_features.shape}")
        
        return text_features

if __name__ == "__main__":
    # Example usage
    text = "slightly surprise"
    embedding = get_text_embedding(text)
    print(f"Embedding shape: {embedding.shape}") 