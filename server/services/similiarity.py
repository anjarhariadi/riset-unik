from model2vec import StaticModel
from fastapi import HTTPException
import numpy as np

from models.schemas import PaperResult

model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")

def encode(texts):
    embeddings = model.encode(texts)
    return embeddings

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
        scores = np.atleast_1d(np.array(sims)).tolist()

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
