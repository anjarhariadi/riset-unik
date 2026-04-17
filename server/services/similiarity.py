from fastapi import HTTPException
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np

from models.schemas import PaperResult

model_dir = "onnx_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
session = InferenceSession(f"{model_dir}/onnx/model_O3.onnx")

def encode(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
    inputs = {k: np.array(v, dtype=np.int64) for k, v in inputs.items()}
    outputs = session.run(None, dict(inputs))
    embeddings = outputs[0]

    mask = np.expand_dims(inputs["attention_mask"], -1)
    mask_sum = np.clip(mask.sum(1), a_min=1e-9, a_max=None)
    mean_emb = (embeddings * mask).sum(1) / mask_sum
    normed = mean_emb / np.linalg.norm(mean_emb, axis=1, keepdims=True)
    return normed

def cosine_similarity(a, b):
    return np.dot(a, b.T).squeeze()

def analyze_similarity(user_topic, titles_with_links):
    if not titles_with_links:
        return []
    
    try:
        titles = [item["title"] for item in titles_with_links if "title" in item and "link" in item]
        if not titles:
            return []

        all_texts = [user_topic] + titles
        all_embeddings = encode(all_texts)
        topic_embedding = all_embeddings[0:1]
        title_embeddings = all_embeddings[1:]

        sims = cosine_similarity(topic_embedding, title_embeddings)
        scores = np.array(sims).tolist()

        results = [
            PaperResult(title=item["title"], link=item["link"], similarity=float(sim))
            for item, sim in zip(titles_with_links, scores)
            if "title" in item and "link" in item
        ]

        results.sort(key=lambda item: item.similarity, reverse=True)
        return results

    except Exception as e:
        print(f"An error occurred during similarity analysis: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat menghitung kemiripan.")
