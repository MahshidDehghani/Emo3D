import pandas as pd
import torch
import clip
from PIL import Image
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def KL(a, b, epsilon):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32) + epsilon
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

class Emo3D_Metric():
    def __init__(self, metric_df_path):
        self.metric_df = pd.read_csv(metric_df_path)
        self.text_tokens, self.text_features = self.get_text_features()
    
    def get_text_features(self):
        text_list = self.metric_df.iloc[:, 0].tolist()
        text_tokens = clip.tokenize(text_list).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        return text_tokens, text_features
    
    def get_image_features(self, image_dir):
        pass

    def get_emotion_distribution(self, index, eps=0):
        emotion = self.metric_df.iloc[index, 1:9].to_numpy().astype(float)
        emotion_eps = emotion + eps
        emotion_dist = emotion_eps / np.sum(emotion_eps)
        return emotion_dist
    

    def get_avg_emotion_distribution(self, indices, eps=0):
        sum_emotions = np.sum(self.metric_df.iloc[indices, 1:9].to_numpy().astype(float), axis=0)
        emotion_eps = sum_emotions + eps
        avg_emotion_dist = emotion_eps / np.sum(emotion_eps)
        return avg_emotion_dist

    def classify_image(self, image_path, k=10):
        image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(k)
        return values.tolist(), indices.tolist()

    def calculate(self, image_dir, alpha, eps=0.01):
        exact_match = 0
        kl_sum = 0
        for i in range(len(self.metric_df)):
            image_path = os.path.join(image_dir, str(i) + ".jpeg")
            image_emotion_dist = self.get_emotion_distribution(i)
            values, indices = self.classify_image(image_path)
            if i in indices:
                exact_match += 1
            text_avg_emotion_dist = self.get_emotion_distribution(indices)
            kl_div = KL(image_emotion_dist, text_avg_emotion_dist, eps)
            kl_sum += kl_div
            print(kl_div)
        exact_match_ratio = exact_match / len(self.metric_df)
        kl_sum_normalized = kl_sum / (len(self.metric_df) * alpha)
        metric_value = 1 / (1 + np.exp(-kl_sum_normalized))
        return exact_match_ratio, kl_sum_normalized, metric_value



            



