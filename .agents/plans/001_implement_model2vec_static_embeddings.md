# Feature: Replace ONNX with Model2Vec Static Embeddings

## Feature Description

Replace the current ONNX-based similarity service with Model2Vec static embeddings to dramatically reduce Docker image size (~829MB → ~200-300MB) and improve inference speed (~500x faster).

## User Story

As a **developer/deployer**
I want to **use Model2Vec static embeddings instead of ONNX**
So that **the Docker image is smaller, inference is faster, and deployment costs are lower**

## Problem Statement

The current ONNX-based similarity service is:

- **829MB Docker image** due to `transformers` library (~400MB) and ONNX model (~90MB)
- **~35ms inference time** per request
- **Complex pooling logic** manually implemented

## Solution Statement

Replace ONNX with Model2Vec:

- **Single library**: `model2vec` (~5MB package)
- **Pre-trained models**: Use `minishlab/potion-base-32M` (English) or `minishlab/potion-multilingual-128M` (Indonesian support)
- **Size**: ~30MB model vs ~90MB ONNX
- **Speed**: Sub-millisecond inference vs ~35ms
- **Simpler code**: Built-in mean pooling

## Feature Metadata

**Feature Type**: Enhancement (Performance Optimization)
**Estimated Complexity**: Low
**Primary Systems Affected**: `server/services/similiarity.py`
**Dependencies**: `model2vec` package

---

## CONTEXT REFERENCES

### Relevant Codebase Files

- `server/services/similiarity.py` (lines 1-55) - **CRITICAL**: Current implementation to replace
- `server/services/similiarity.py` (lines 24-25) - `cosine_similarity` function to keep
- `server/models/schemas.py` (lines 7-10) - `PaperResult` schema (unchanged)
- `server/requirements.txt` (lines 1-10) - Dependencies to update
- `server/export_to_onnx.py` - File to delete (no longer needed)
- `server/Dockerfile` (lines 1-16) - May need cleanup of ONNX model copy

### New Files to Create

- None (Model2Vec downloads model automatically)

### Files to Delete

- `server/onnx_model/` - Entire directory (no longer needed)
- `server/export_to_onnx.py` - No longer needed

---

## Relevant Documentation

- [Model2Vec GitHub](https://github.com/MinishLab/model2vec)
  - Quickstart guide
  - Why: Core library documentation
- [Model2Vec Introduction](https://minish.ai/packages/model2vec/introduction)
  - Installation and basic usage
  - Why: Official getting started guide
- [Model2Vec HuggingFace Models](https://huggingface.co/MinishLab)
  - Available pre-trained models
  - Why: Model selection reference

### Model Selection

| Model                                | Languages                        | Size   | Use Case                        |
| ------------------------------------ | -------------------------------- | ------ | ------------------------------- |
| `minishlab/potion-base-32M`          | English                          | ~30MB  | English-only (current behavior) |
| `minishlab/potion-multilingual-128M` | 101 languages (incl. Indonesian) | ~128MB | Better for Indonesian text      |

**Recommendation**: Use `minishlab/potion-multilingual-128M` since RisetUnik serves Indonesian users.

---

## Patterns to Follow

### Current Pattern (similarity.py)

```python
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np

model_dir = "onnx_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
session = InferenceSession(f"{model_dir}/onnx/model_O3.onnx")

def encode(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
    inputs = {k: np.array(v, dtype=np.int64) for k, v in inputs.items()}
    outputs = session.run(None, dict(inputs))
    embeddings = outputs[0]
    # Manual mean pooling
    mask = np.expand_dims(inputs["attention_mask"], -1)
    mask_sum = np.clip(mask.sum(1), a_min=1e-9, a_max=None)
    mean_emb = (embeddings * mask).sum(1) / mask_sum
    normed = mean_emb / np.linalg.norm(mean_emb, axis=1, keepdims=True)
    return normed

def cosine_similarity(a, b):
    return np.dot(a, b.T).squeeze()

def analyze_similarity(user_topic, titles_with_links):
    # ... implementation
```

### New Pattern (Model2Vec)

```python
from model2vec import StaticModel
import numpy as np

# Load model at startup
model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")

def encode(texts):
    # Returns already normalized embeddings
    embeddings = model.encode(texts)
    return embeddings

def cosine_similarity(a, b):
    return np.dot(a, b.T).squeeze()

def analyze_similarity(user_topic, titles_with_links):
    # ... same logic, just cleaner
```

---

## IMPLEMENTATION PLAN

### Phase 1: Update Dependencies

**Tasks:**

- Replace `transformers` with `model2vec` in requirements.txt
- Remove `onnxruntime` if no longer needed

### Phase 2: Rewrite Similarity Service

**Tasks:**

- Replace tokenizer + ONNX session with Model2Vec StaticModel
- Remove manual pooling logic (Model2Vec handles this)
- Keep cosine_similarity function unchanged
- Keep analyze_similarity function signature unchanged

### Phase 3: Cleanup

**Tasks:**

- Delete `export_to_onnx.py` (no longer needed)
- Delete `onnx_model/` directory (no longer needed)
- Update `.dockerignore` to remove ONNX references

---

## STEP-BY-STEP TASKS

### UPDATE server/requirements.txt

- **REMOVE**: `transformers`
- **REMOVE**: `onnxruntime`
- **ADD**: `model2vec`
- **PATTERN**: See `server/requirements.txt:1-10`

```
# Before
transformers
onnxruntime

# After
model2vec
```

---

### REFACTOR server/services/similarity.py

- **REPLACE** entire file content with Model2Vec implementation
- **PATTERN**: See Patterns section above
- **IMPORTS**: `from model2vec import StaticModel`
- **MODEL**: Use `"minishlab/potion-multilingual-128M"` for Indonesian support
- **KEEP**: `cosine_similarity` function unchanged
- **KEEP**: `PaperResult` import (schema unchanged)
- **KEEP**: `analyze_similarity` function signature and logic
- **SIMPLIFY**: Remove manual pooling (Model2Vec handles it)
- **VALIDATE**: `python -c "from services.similiarity import analyze_similarity; print('OK')"`

```python
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
```

---

### DELETE server/export_to_onnx.py

- **ACTION**: Remove file entirely
- **REASON**: No longer needed for model export
- **VALIDATE**: `ls server/export_to_onnx.py` should fail

---

### DELETE server/onnx_model/ directory

- **ACTION**: Remove entire directory recursively
- **REASON**: ONNX model no longer needed
- **VALIDATE**: `ls server/onnx_model/` should fail

---

### UPDATE server/.dockerignore

- **REMOVE** any references to `onnx_model`, `model`, `onnx`
- **PATTERN**: See `server/.dockerignore:1-12`
- **VALIDATE**: `grep -i onnx server/.dockerignore` should return nothing

---

### UPDATE server/Dockerfile

- **REMOVE**: ONNX-specific dependencies if any remain
- **SIMPLIFY**: Remove any ONNX-related comments or configurations
- **PATTERN**: Keep minimal Python slim setup
- **VALIDATE**: Build image and verify

---

## TESTING STRATEGY

### Unit Tests

1. **Import Test**: Verify `analyze_similarity` can be imported
2. **Empty Input Test**: Verify empty titles returns empty list
3. **Basic Similarity Test**: Run with sample titles and verify results
4. **Score Range Test**: Verify all similarity scores are between -1 and 1
5. **Sorting Test**: Verify results are sorted by similarity descending

### Integration Tests

1. **API Endpoint Test**: Call `/analyze-topic` with sample topic
2. **Yapping Mode Test**: Call `/analyze-yapping` with sample description
3. **Concurrent Requests**: Test multiple simultaneous requests

### Edge Cases

1. Empty topic string
2. Very long topic strings
3. Non-English text (Indonesian)
4. Special characters in text
5. Empty titles list
6. Titles with missing fields

---

## VALIDATION COMMANDS

### Level 1: Syntax & Import

```bash
cd server && python -c "from services.similiarity import analyze_similarity, encode; print('Import OK')"
```

### Level 2: Unit Tests

```bash
cd server && python -c "
from services.similiarity import analyze_similarity

# Test 1: Empty input
result = analyze_similarity('test', [])
assert result == [], 'Empty test failed'
print('Test 1: Empty input - PASS')

# Test 2: Basic similarity
titles = [
    {'title': 'Deep Learning for NLP', 'link': 'http://example.com/1'},
    {'title': 'Computer Vision Survey', 'link': 'http://example.com/2'},
]
result = analyze_similarity('Machine learning for text', titles)
assert len(result) == 2, 'Basic test failed'
assert all(-1 <= r.similarity <= 1 for r in result), 'Score range failed'
assert result[0].similarity >= result[1].similarity, 'Sort failed'
print('Test 2: Basic similarity - PASS')

# Test 3: Scores are reasonable (NLP topic should match NLP paper higher)
nlp_result = [r for r in result if 'NLP' in r.title][0]
cv_result = [r for r in result if 'Vision' in r.title][0]
assert nlp_result.similarity > cv_result.similarity, 'Relevance test failed'
print('Test 3: Relevance ranking - PASS')

print('All tests passed!')
"
```

### Level 3: API Integration

```bash
# Start server (requires API keys in .env)
cd server && uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 3

# Test endpoint
curl -X POST http://localhost:8000/analyze-topic \
  -H "Content-Type: application/json" \
  -d '{"topic": "Machine learning for NLP"}'

# Kill server
pkill -f "uvicorn main:app"
```

### Level 4: Docker Build

```bash
# Build new image
cd server && docker build -t risetunik-server-test .

# Check size
docker images risetunik-server-test

# Run container
docker run -d --name risetunik-test -p 8001:8000 risetunik-server-test

# Test
curl http://localhost:8001/

# Cleanup
docker stop risetunik-test && docker rm risetunik-test
```

---

## ACCEPTANCE CRITERIA

- [ ] `model2vec` replaces `transformers` and `onnxruntime` in requirements.txt
- [ ] `server/services/similarity.py` uses Model2Vec with <50 lines
- [ ] `server/export_to_onnx.py` is deleted
- [ ] `server/onnx_model/` directory is deleted
- [ ] Docker image size is reduced to <300MB (from 829MB)
- [ ] Inference time is <1ms per batch (from ~35ms)
- [ ] All unit tests pass
- [ ] API endpoint `/analyze-topic` works correctly
- [ ] API endpoint `/analyze-yapping` works correctly
- [ ] Indonesian text produces valid similarity scores
- [ ] Results are correctly sorted by similarity

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Unit tests pass
- [ ] API tests pass
- [ ] Docker image builds successfully
- [ ] Docker image size <300MB
- [ ] No references to ONNX remain in codebase
- [ ] Code is cleaner than before (fewer lines)
- [ ] Functionality is identical to before

---

## NOTES

### Why Multilingual Model?

RisetUnik serves Indonesian users who write topics in Indonesian. The current model (`paraphrase-MiniLM-L6-v2`) is English-only. Using `potion-multilingual-128M` will:

- Better handle Indonesian text
- Maintain English support
- Still be much smaller than ONNX model

### Performance Expectations

| Metric          | ONNX (Current) | Model2Vec (New) | Improvement      |
| --------------- | -------------- | --------------- | ---------------- |
| Docker Image    | ~829MB         | ~200-300MB      | **3-4x smaller** |
| Inference Time  | ~35ms          | <1ms            | **35x faster**   |
| Code Complexity | 55 lines       | ~45 lines       | Simpler          |
| Model Download  | Pre-bundled    | On-first-run    | Smaller image    |

### First-Run Behavior

Model2Vec downloads the model on first run (if not cached). This means:

- First request may take longer (model download)
- Subsequent requests are instant
- Can pre-download model during Docker build if needed

### Alternative: Distill Custom Model

If you want to preserve exact quality of current model:

```bash
pip install "model2vec[distill]"
python -c "
from model2vec.distill import distill
m2v = distill('sentence-transformers/paraphrase-MiniLM-L6-v2')
m2v.save_pretrained('custom_model')
"
```

This would take ~30 seconds on CPU and create a custom Model2Vec model trained from current model.
