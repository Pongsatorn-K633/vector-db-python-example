# Vector Database Python - Multi-Modal Search Lab

This project demonstrates various vector database implementations for semantic search, featuring both text and image retrieval using modern embedding models.

## 📚 Overview

The project consists of three main notebooks showcasing different vector database approaches:

1. **SQLite with Vector Search** - Using `sqlite-vec` for lightweight local vector search
2. **Chroma Text Search** - Text-based semantic similarity with Chroma
3. **Chroma Image Search** - Multi-modal image retrieval with CLIP embeddings

## 🚀 Project Structure

```
vector-db-python/
├── 01_sqlite.ipynb           # SQLite vector extension example
├── 02_chroma_text.ipynb      # Chroma text semantic search
├── 03_chroma_image.ipynb     # Chroma image search with CLIP
├── images/                   # Sample images for testing
├── Readme.md
└── demo.db                   # SQLite database (generated)
```

## 📖 Detailed Notebooks

### 1. SQLite Vector Search (`01_sqlite.ipynb`)

**Description**: Lightweight vector search using SQLite with the `sqlite-vec` extension.

**Key Features**:
- Uses SQLite's virtual table extension `vec0` for efficient vector operations
- Deterministic embeddings based on SHA256 hashing
- L2 distance-based similarity search
- Minimal dependencies for local-first applications

**Workflow**:
1. Create SQLite connection and load `sqlite-vec` extension
2. Create `docs` table with sample text documents
3. Generate 8-dimensional embeddings for each document
4. Create `vec_docs` virtual table for vector indexing
5. Query similar documents using distance-based search

**Sample Output**:
```
- rowid=1  distance=0.385928...
- rowid=2  distance=0.420892...
```

**Use Cases**:
- Small-scale semantic search with minimal setup
- Desktop applications requiring vector search
- Prototyping vector search without external services

---

### 2. Chroma Text Search (`02_chroma_text.ipynb`)

**Description**: Text-based semantic similarity search using Chroma vector database.

**Key Features**:
- Uses Chroma's built-in embedding model
- Automatic text embedding and indexing
- Simple document collection management
- Cosine similarity-based retrieval

**Workflow**:
1. Initialize Chroma client
2. Create a collection named `demo_collection`
3. Add text documents with IDs
4. Query with natural language text
5. Retrieve results with similarity distances

**Query Example**:
```
Query: "fox dog"
- distance=0.123456  content="The quick brown fox jumps over the lazy dog"
- distance=0.234567  content="A fast auburn fox leaps above a sleepy canine"
```

**Use Cases**:
- FAQ systems and document retrieval
- Semantic search in knowledge bases
- Quick prototyping without model configuration

---

### 3. Chroma Image Search (`03_chroma_image.ipynb`)

**Description**: Multi-modal image retrieval using OpenCLIP embeddings with Chroma's built-in data loaders.

**Key Features**:
- Uses `OpenCLIPEmbeddingFunction` for automatic vision embeddings
- Automatic image loading with `ImageLoader` data loader
- Persistent local storage with `PersistentClient`
- Cosine distance metric (HNSW space)
- Rich metadata tracking (filename, label, description)
- Text-to-image search with automatic embedding

**Workflow**:
1. Import ChromaDB embedding functions and data loaders
2. Configure paths and image file lists (dogs and cats)
3. Prepare image URIs and metadata (label, filename, description)
4. Initialize `PersistentClient` for persistent storage
5. Create collection with `OpenCLIPEmbeddingFunction` and `ImageLoader`
6. Add images using URIs (automatic embedding and loading)
7. Query images by text (automatic text embedding)

**Key Parameters**:
- `OpenCLIPEmbeddingFunction()` - Handles image embedding automatically
- `ImageLoader()` - Loads images from file paths automatically
- `data_loader=ImageLoader()` - Tells ChromaDB to load images from URIs
- `query_texts=[...]` - Send text queries directly, no manual embedding needed

**Query Example**:
```
=== Query: 'dog' ===
#1 -> dog1.png (dog) distance=0.0234
  desc: A photo of a dog
#2 -> dog3.png (dog) distance=0.0456
  desc: A photo of a dog
#3 -> dog2.png (dog) distance=0.0789
  desc: A photo of a dog
#4 -> cat1.png (cat) distance=0.3421
  desc: A photo of a cat
```

**Use Cases**:
- Product image search
- Content-based image retrieval with text queries
- Multi-modal recommendation systems
- Automatic image indexing without manual preprocessing

---

## 🛠 Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Install Dependencies

```bash
pip install chromadb sentence-transformers pillow sqlite-vec numpy
```

### Notebook-specific Requirements

**For 01_sqlite.ipynb**:
```bash
pip install sqlite-vec
```

**For 02_chroma_text.ipynb**:
```bash
pip install chromadb
```

**For 03_chroma_image.ipynb**:
```bash
pip install chromadb open-clip-torch
```

---

## 📊 Comparison

| Feature | SQLite | Chroma Text | Chroma Image |
|---------|--------|-------------|--------------|
| **Model** | SHA256 Hash | Default (Chroma) | CLIP ViT-B-32 |
| **Input** | Text | Text | Images |
| **Persistence** | File-based DB | Optional | Local folder |
| **Setup Complexity** | Simple | Easy | Medium |
| **Scalability** | Small-Medium | Medium-Large | Medium-Large |
| **Dependencies** | sqlite-vec | chromadb | chromadb + transformers |

---

## 🎯 Next Steps

- Experiment with different embedding models in Chroma
- Scale image search to larger datasets
- Implement filtering with metadata queries
- Combine text and image search in a hybrid system
- Deploy as a web service with FastAPI

---

## 📝 Notes

- Images for `03_chroma_image.ipynb` should be placed in the `images/` directory
- Chroma collections use cosine distance by default
- CLIP embeddings work well for cross-modal (text→image, image→image) retrieval
- SQLite vector search is best for small datasets with offline requirements

---

## 📜 License

Educational and experimental project for learning vector database concepts.
