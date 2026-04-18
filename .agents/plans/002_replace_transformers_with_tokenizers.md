# Feature: Replace transformers with tokenizers Library

## Feature Description

Replace the `transformers` library with the standalone `tokenizers` library to reduce Docker image size. The `transformers` library (~400MB) pulls in PyTorch and other heavy dependencies just for tokenizer functionality. The `tokenizers` library (~5MB) provides the same tokenizer API without the overhead.

## User Story

As a **developer/deployer**
I want to **replace transformers with just the tokenizers library**
So that **the Docker image is smaller while maintaining identical functionality**

## Problem Statement

The current `transformers` library is installed only for:
- `AutoTokenizer` to load the tokenizer
- This pulls in PyTorch and other heavy dependencies (~400MB)
- Docker image is ~829MB

## Solution Statement

Replace `transformers.AutoTokenizer` with `tokenizers.Tokenizer`:
- `tokenizers` library is ~5MB (standalone, no PyTorch dependency)
- Same tokenizer functionality for ONNX inference
- Docker image can be reduced by ~400MB

## Feature Metadata

**Feature Type**: Enhancement (Performance Optimization)
**Estimated Complexity**: Low
**Primary Systems Affected**: `server/services/similiarity.py`
**Dependencies**: Replace `transformers` with `tokenizers`

---

## CONTEXT REFERENCES

### Relevant Codebase Files

- `server/services/similiarity.py` (lines 1-55) - **CRITICAL**: Replace AutoTokenizer import
- `server/services/similiarity.py` (lines 12-22) - `encode` function to update
- `server/services/similiarity.py` (lines 24-25) - `cosine_similarity` function (unchanged)
- `server/services/similiarity.py` (lines 27-55) - `analyze_similarity` function (unchanged)
- `server/onnx_model/tokenizer.json` - Tokenizer configuration file (already exists)
- `server/requirements.txt` (lines 1-10) - Dependencies to update

### Current Implementation

```python
# Line 2
from transformers import AutoTokenizer

# Line 9
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

# Line 13-14
def encode(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
    inputs = {k: np.array(v, dtype=np.int64) for k, v in inputs.items()}
```

---

## Relevant Documentation

- [Tokenizers Library Documentation](https://huggingface.co/docs/tokenizers/main/en/api/tokenizer)
  - Tokenizer class API
  - Why: Core library documentation for replacement
- [Tokenizers from_pretrained](https://huggingface.co/docs/tokenizers/main/en/quicktour#loading-a-tokenizer)
  - Loading tokenizers from local files
  - Why: Shows how to load from local directory

---

## Patterns to Follow

### Current Pattern (transformers)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/model", local_files_only=True)

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
# inputs = {"input_ids": np.array, "attention_mask": np.array}
inputs = {k: np.array(v, dtype=np.int64) for k, v in inputs.items()}
```

### New Pattern (tokenizers)

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("path/to/model")  # Loads tokenizer.json

# Encode batch
encodings = tokenizer.encode_batch(texts)

# Extract ids and attention_mask
ids = [e.ids for e in encodings]
masks = [e.attention_mask for e in encodings]

# Pad manually (tokenizers doesn't auto-pad by default)
max_len = max(len(x) for x in ids)
ids = [x + [0] * (max_len - len(x)) for x in ids]
masks = [x + [0] * (max_len - len(x)) for x in masks]

inputs = {
    "input_ids": np.array(ids, dtype=np.int64),
    "attention_mask": np.array(masks, dtype=np.int64)
}
```

---

## IMPORTANT: Encoding Format Differences

### transformers AutoTokenizer
```python
tokenizer(texts, padding=True, truncation=True, return_tensors="np")
# Returns: {"input_ids": np.array(...), "attention_mask": np.array(...)}
```

### tokenizers Tokenizer
```python
tokenizer.encode_batch(texts)
# Returns: List[Encoding] where Encoding has:
#   - .ids: List[int]
#   - .attention_mask: List[int]
#   - .type_ids: List[int]
#   - etc.

# Key differences:
# 1. No automatic padding - must pad manually
# 2. No truncation by default - must set max_length
# 3. Returns list of Encoding objects, not dict
```

---

## IMPLEMENTATION PLAN

### Phase 1: Update Dependencies

**Tasks:**
- Replace `transformers` with `tokenizers` in requirements.txt

### Phase 2: Rewrite Similarity Service

**Tasks:**
- Replace `AutoTokenizer` with `Tokenizer`
- Update `encode` function to handle tokenizers API differences
- Keep ONNX inference logic unchanged

---

## STEP-BY-STEP TASKS

### UPDATE server/requirements.txt

- **REMOVE**: `transformers`
- **ADD**: `tokenizers`
- **PATTERN**: See `server/requirements.txt:1-10`

```
# Before
transformers

# After
tokenizers
```

---

### REFACTOR server/services/similiarity.py

**Task 1: Replace imports**

- **REMOVE** line 2: `from transformers import AutoTokenizer`
- **ADD** after imports: `from tokenizers import Tokenizer`

**Task 2: Replace tokenizer initialization**

- **REMOVE** line 9: `tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)`
- **ADD**: `tokenizer = Tokenizer.from_pretrained(model_dir)`

**Task 3: Update encode function**

- **REPLACE** the entire `encode` function
- **PATTERN**: See Patterns section above
- **GOTCHA**: tokenizers doesn't auto-pad by default, must pad manually
- **GOTCHA**: tokenizers doesn't auto-truncate, must set `max_length` parameter

**Complete new encode function:**

```python
def encode(texts):
    # Encode batch
    encodings = tokenizer.encode_batch(texts)
    
    # Extract ids and masks
    ids = [e.ids for e in encodings]
    masks = [e.attention_mask for e in encodings]
    
    # Manual padding to max length
    max_len = max(len(x) for x in ids)
    padded_ids = [x + [0] * (max_len - len(x)) for x in ids]
    padded_masks = [x + [0] * (max_len - len(x)) for x in masks]
    
    inputs = {
        "input_ids": np.array(padded_ids, dtype=np.int64),
        "attention_mask": np.array(padded_masks, dtype=np.int64)
    }
    
    outputs = session.run(None, inputs)
    embeddings = outputs[0]
    
    # Mean pooling
    mask = np.expand_dims(inputs["attention_mask"], -1)
    mask_sum = np.clip(mask.sum(1), a_min=1e-9, a_max=None)
    mean_emb = (embeddings * mask).sum(1) / mask_sum
    normed = mean_emb / np.linalg.norm(mean_emb, axis=1, keepdims=True)
    return normed
```

**KEEP** `cosine_similarity` function unchanged (lines 24-25)

**KEEP** `analyze_similarity` function unchanged (lines 27-55)

---

### VALIDATE

```bash
cd server && python -c "from services.similiarity import analyze_similarity; print('Import OK')"
```

---

## TESTING STRATEGY

### Unit Tests

1. **Import Test**: Verify module imports without errors
2. **Empty Input Test**: Empty titles returns empty list
3. **Basic Similarity Test**: Run with sample titles
4. **Score Range Test**: All scores between -1 and 1
5. **Sorting Test**: Results sorted by similarity descending

### Integration Tests

1. **API Endpoint Test**: `/analyze-topic` works
2. **Yapping Mode Test**: `/analyze-yapping` works

### Edge Cases

1. Empty topic string
2. Very long text
3. Special characters
4. Empty titles list
5. Missing fields in titles

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

print('All tests passed!')
"
```

### Level 3: Docker Build

```bash
cd server && docker build -t risetunik-server-test .

# Check size
docker images risetunik-server-test

# Compare sizes
docker images | grep risetunik
```

---

## ACCEPTANCE CRITERIA

- [ ] `transformers` removed from requirements.txt
- [ ] `tokenizers` added to requirements.txt
- [ ] `services/similarity.py` uses `tokenizers.Tokenizer` instead of `transformers.AutoTokenizer`
- [ ] Encode function handles manual padding
- [ ] All unit tests pass
- [ ] API endpoints work correctly
- [ ] Docker image size is reduced (target: <500MB from 829MB)

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Unit tests pass
- [ ] API tests pass
- [ ] Docker image builds successfully
- [ ] Docker image size reduced by ~400MB

---

## NOTES

### Why This Works

The ONNX model only needs:
1. Tokenized inputs (`input_ids`, `attention_mask`)
2. These are NumPy arrays (already supported)

The `tokenizers` library provides the same tokenization without PyTorch dependency.

### Trade-offs

| Aspect | transformers | tokenizers |
|--------|--------------|------------|
| Size | ~400MB | ~5MB |
| Auto-padding | Yes | No (manual) |
| Auto-truncation | Yes | No (manual) |
| Model loading | AutoTokenizer | Tokenizer |
| ONNX compatibility | 100% | 100% |

### Size Comparison

| Component | Current | After |
|-----------|---------|-------|
| transformers | ~400MB | 0 |
| tokenizers | 0 | ~5MB |
| Net change | - | **~395MB smaller** |

### Alternative: HuggingFace Tokenizers

You can also use `huggingface_hub` to download tokenizer from HuggingFace:

```python
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

tokenizer_file = hf_hub_download(repo_id="path/to/model", filename="tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_file)
```

But since tokenizer.json is already in the repo, loading locally is preferred.
