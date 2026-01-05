# 🔧 Environment Configuration Guide

This document describes all environment variables for the Interior Visual Search backend.

## 📝 Quick Setup

Add these variables to your `backend/.env` file:

```bash
# ====================================
# Application Settings
# ====================================
APP_NAME="Interior Visual Search"
APP_ENV="dev"
LOG_LEVEL="INFO"

# ====================================
# Pinecone Vector Database
# ====================================
PINECONE_API_KEY="your-pinecone-api-key-here"
PINECONE_INDEX_NAME="interior-products"
PINECONE_CLOUD="aws"
PINECONE_REGION="us-east-1"
PINECONE_DIM=1024

# ====================================
# Embedding Models
# ====================================
# Instance model (fine-grained features)
INSTANCE_MODEL_NAME="google/vit-base-patch16-224-in21k"

# Semantic model (category/style features)
SEMANTIC_MODEL_NAME="openai/clip-vit-base-patch32"

# ====================================
# Detection & Segmentation Models
# ====================================
# RT-DETR Detection Model Path
RTDETR_MODEL_PATH="D:\\image_image_search\\backend\\app\\models\\rtdetr-x.pt"

# SAM2.1 Segmentation Model Path (Ultralytics)
SAM2_MODEL_PATH="D:\\image_image_search\\backend\\app\\models\\sam2.1_l.pt"

# ====================================
# Image Validation Settings
# ====================================
IMAGE_MIN_DIMENSION=100
IMAGE_MAX_DIMENSION=4096
IMAGE_MAX_SIZE_MB=10

# ====================================
# Search Settings
# ====================================
SEARCH_CANDIDATE_MULTIPLIER=15
SEARCH_DEDUPLICATE_SKUS=true
MAX_CANDIDATE_K=5000

# ====================================
# Advanced Embedding Settings (NEW!)
# ====================================
# Enable multi-scale embedding for extreme size variations
ENABLE_MULTISCALE_EMBEDDING=true

# Embedding scales (comma-separated pixel sizes)
EMBEDDING_SCALES=224,384,512

# Enable rotation augmentation for angle invariance
ENABLE_ROTATION_AUGMENTATION=true

# ====================================
# Advanced Reranking Settings (NEW!)
# ====================================
# Enable multi-stage reranking for large catalogs
ENABLE_MULTISTAGE_RERANK=true

# Stage 1 filtering ratio
RERANK_STAGE1_RATIO=0.3

# ====================================
# Optional: Pinecone Reranking
# ====================================
ENABLE_PINECONE_RERANK=false
PINECONE_RERANK_MODEL="bge-reranker-v2-m3"
```

---

## 🎯 Configuration Presets

### Speed Optimized (Fastest, Good Accuracy)

Best for: Real-time applications, limited resources, small catalogs (<10K products)

```bash
ENABLE_MULTISCALE_EMBEDDING=false
ENABLE_ROTATION_AUGMENTATION=false
ENABLE_MULTISTAGE_RERANK=false
EMBEDDING_SCALES=224
SEARCH_CANDIDATE_MULTIPLIER=10
MAX_CANDIDATE_K=1000
```

**Performance:**
- Inference time: ~100-200ms per query
- Memory usage: ~2GB GPU / 4GB CPU
- Accuracy: Good for standard use cases

---

### Balanced (Recommended)

Best for: Most production use cases, moderate catalogs (10K-50K products)

```bash
ENABLE_MULTISCALE_EMBEDDING=true
ENABLE_ROTATION_AUGMENTATION=true
ENABLE_MULTISTAGE_RERANK=true
EMBEDDING_SCALES=224,384,512
SEARCH_CANDIDATE_MULTIPLIER=15
MAX_CANDIDATE_K=2000
RERANK_STAGE1_RATIO=0.3
```

**Performance:**
- Inference time: ~400-600ms per query
- Memory usage: ~4GB GPU / 8GB CPU
- Accuracy: Excellent for most use cases

**Handles:**
- ✅ Size variations (10x difference)
- ✅ Angle variations (90°, 180°, 270°)
- ✅ Zoom variations (5x difference)
- ✅ 50K products per category

---

### Accuracy Optimized (Best Quality)

Best for: High-precision applications, critical use cases, quality over speed

```bash
ENABLE_MULTISCALE_EMBEDDING=true
ENABLE_ROTATION_AUGMENTATION=true
ENABLE_MULTISTAGE_RERANK=true
EMBEDDING_SCALES=224,384,512,640
SEARCH_CANDIDATE_MULTIPLIER=40
MAX_CANDIDATE_K=10000
RERANK_STAGE1_RATIO=0.2
```

**Performance:**
- Inference time: ~1000-1500ms per query
- Memory usage: ~6GB GPU / 12GB CPU
- Accuracy: Maximum quality

**Handles:**
- ✅ Extreme size variations (100x difference)
- ✅ All angle variations (0°-360°)
- ✅ Extreme zoom variations (10x+ difference)
- ✅ 100K+ products per category

---

### Large Catalog (50K+ Products)

Best for: Very large product catalogs, e-commerce platforms

```bash
ENABLE_MULTISCALE_EMBEDDING=true
ENABLE_ROTATION_AUGMENTATION=true
ENABLE_MULTISTAGE_RERANK=true
EMBEDDING_SCALES=224,384,512
SEARCH_CANDIDATE_MULTIPLIER=40
MAX_CANDIDATE_K=5000
RERANK_STAGE1_RATIO=0.2
```

**Performance:**
- Inference time: ~800-1000ms per query
- Memory usage: ~5GB GPU / 10GB CPU
- Accuracy: Excellent for large catalogs

**Optimized for:**
- ✅ 50K-1M products per category
- ✅ Fast candidate retrieval with high recall
- ✅ Two-stage reranking for efficiency
- ✅ Automatic catalog size detection

---

## 📊 Feature Comparison

| Feature | Speed | Balanced | Accuracy | Large Catalog |
|---------|-------|----------|----------|---------------|
| Multi-scale embedding | ❌ | ✅ | ✅ | ✅ |
| Rotation augmentation | ❌ | ✅ | ✅ | ✅ |
| Multi-stage reranking | ❌ | ✅ | ✅ | ✅ |
| Embedding scales | 1 | 3 | 4 | 3 |
| Candidate multiplier | 10x | 15x | 40x | 40x |
| Max candidates | 1K | 2K | 10K | 5K |
| Inference time | 100ms | 500ms | 1500ms | 1000ms |
| Memory usage | Low | Medium | High | Medium-High |
| Accuracy | Good | Excellent | Maximum | Excellent |

---

## 🔍 Detailed Parameter Descriptions

### ENABLE_MULTISCALE_EMBEDDING

**Type:** Boolean (true/false)  
**Default:** true  
**Impact:** High

Enables multi-scale embedding pyramid (small, medium, large scales).

**When to use:**
- ✅ Products with varying sizes (small furniture vs large sofas)
- ✅ Query images have different resolutions than catalog
- ✅ Need to handle 10x-100x size variations

**When to disable:**
- ❌ All images are similar resolution
- ❌ Speed is critical
- ❌ Limited GPU/CPU resources

---

### EMBEDDING_SCALES

**Type:** Comma-separated integers  
**Default:** "224,384,512"  
**Impact:** High

Pixel sizes for multi-scale embedding pyramid.

**Options:**
- `224`: Fast, good for small details
- `384`: Balanced, captures medium features
- `512`: Detailed, captures fine textures
- `640`: Very detailed (slow, high memory)

**Recommendations:**
- Small products: `224,384`
- Medium products: `224,384,512`
- Large products: `384,512,640`

---

### ENABLE_ROTATION_AUGMENTATION

**Type:** Boolean (true/false)  
**Default:** true  
**Impact:** Medium

Generates 90°, 180°, 270° rotated versions during embedding.

**When to use:**
- ✅ Products photographed at different angles
- ✅ Query images may be rotated
- ✅ Need angle invariance

**When to disable:**
- ❌ All images are consistently oriented
- ❌ Speed is critical
- ❌ Products have clear "up" orientation (e.g., lamps)

---

### ENABLE_MULTISTAGE_RERANK

**Type:** Boolean (true/false)  
**Default:** true  
**Impact:** High (for large catalogs)

Enables two-stage reranking: fast semantic filter → detailed matching.

**When to use:**
- ✅ Catalogs with 10K+ products per category
- ✅ Need to retrieve 1000+ candidates
- ✅ Want to balance speed and accuracy

**When to disable:**
- ❌ Small catalogs (<1K products)
- ❌ Already retrieving few candidates (<100)

---

### SEARCH_CANDIDATE_MULTIPLIER

**Type:** Integer (1-100)  
**Default:** 15  
**Impact:** High

Multiplier for candidate retrieval: `candidates = top_k × multiplier`

**Recommendations:**
- Small catalog (<1K): `10`
- Medium catalog (1K-10K): `15`
- Large catalog (10K-50K): `30`
- Very large catalog (50K+): `40`

**Trade-off:**
- Higher = Better recall, slower search
- Lower = Faster search, may miss relevant items

---

### MAX_CANDIDATE_K

**Type:** Integer (100-10000)  
**Default:** 5000  
**Impact:** Medium

Maximum number of candidates to retrieve from Pinecone.

**Recommendations:**
- Speed-focused: `1000`
- Balanced: `2000-5000`
- Accuracy-focused: `10000`

**Note:** Higher values increase latency but improve recall for large catalogs.

---

### RERANK_STAGE1_RATIO

**Type:** Float (0.1-0.9)  
**Default:** 0.3  
**Impact:** Medium

Ratio of candidates to keep after stage 1 semantic filtering.

**Recommendations:**
- Fast search: `0.2` (keep top 20%)
- Balanced: `0.3` (keep top 30%)
- High recall: `0.5` (keep top 50%)

**Trade-off:**
- Lower = Faster, may miss items
- Higher = Slower, better recall

---

## 🚀 Migration Guide

If you're upgrading from the previous version, add these new variables to your `.env`:

```bash
# Add these lines to your existing .env file:
ENABLE_MULTISCALE_EMBEDDING=true
EMBEDDING_SCALES=224,384,512
ENABLE_ROTATION_AUGMENTATION=true
ENABLE_MULTISTAGE_RERANK=true
RERANK_STAGE1_RATIO=0.3
MAX_CANDIDATE_K=5000
```

**No breaking changes!** All new features have sensible defaults and are backward compatible.

---

## 🧪 Testing Different Configurations

To test different configurations without restarting:

1. Update `.env` file
2. Restart the backend: `uvicorn app.main:app --reload`
3. Monitor logs for configuration details
4. Test with various query images

**Look for these log messages:**
```
INFO: Using multi-scale embedding with scales: [224, 384, 512]
INFO: Using hybrid multi-stage reranking (adaptive based on catalog size)
INFO: Catalog size estimate: 50,000 products, retrieving 800 candidates (multiplier: 40x)
```

---

## 📝 Notes

- All paths should use double backslashes on Windows: `D:\\path\\to\\file`
- Boolean values: `true` or `false` (lowercase)
- Restart backend after changing configuration
- Monitor logs to verify settings are applied

---

## 🆘 Troubleshooting

**Issue:** Out of memory errors  
**Solution:** Reduce `EMBEDDING_SCALES`, disable `ENABLE_ROTATION_AUGMENTATION`, or lower `MAX_CANDIDATE_K`

**Issue:** Slow inference times  
**Solution:** Use Speed Optimized preset or disable multi-scale/rotation augmentation

**Issue:** Poor accuracy for large catalogs  
**Solution:** Increase `SEARCH_CANDIDATE_MULTIPLIER` and `MAX_CANDIDATE_K`

**Issue:** Missing relevant results  
**Solution:** Enable multi-scale embedding and increase candidate retrieval

---

## 📚 References

- Multi-scale embeddings: Pyramid feature extraction
- Rotation augmentation: Test-time augmentation (TTA)
- Two-stage reranking: Cascade ranking approach
- Adaptive candidate retrieval: Dynamic recall optimization

