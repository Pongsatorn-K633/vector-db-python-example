from __future__ import annotations
import os, uuid
from typing import List
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------- Config ----------------
PERSIST_DIR = "./chroma_multimodal_local"
COLLECTION_NAME = "pets_local"

DOG_IMAGES = ["images/dog1.png", "images/dog2.png", "images/dog3.png"]
CAT_IMAGES = ["images/cat1.png", "images/cat2.png", "images/cat3.png"]

# ---------------- Helpers ----------------
def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def load_images(paths: List[str]) -> List[Image.Image]:
    """โหลดรูปจาก path ที่ให้มา"""
    return [Image.open(path).convert("RGB") for path in paths]

def encode_images(model: SentenceTransformer, images: List[Image.Image]) -> np.ndarray:
    """แปลงรูปเป็น embedding แล้ว normalize"""
    embs = model.encode(images, convert_to_numpy=True, batch_size=4, normalize_embeddings=False)
    return l2_normalize(embs)

def setup_collection(client: chromadb.Client, name: str):
    """สร้าง collection ถ้ายังไม่มี"""
    try:
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception:
        return client.get_collection(name=name)

def add_images_to_db(collection, images: List[str], labels: List[str], embs: np.ndarray):
    """เพิ่มรูปและ metadata เข้า Chroma"""
    doc_ids = [str(uuid.uuid4()) for _ in images]
    metadatas = [{"label": lbl, "filename": os.path.basename(path)} for lbl, path in zip(labels, images)]
    documents = [f"A {lbl} image ({os.path.basename(path)})" for lbl, path in zip(labels, images)]

    collection.add(
        ids=doc_ids,
        embeddings=embs.tolist(),
        metadatas=metadatas,
        documents=documents
    )
    print(f"Indexed {len(doc_ids)} images into collection '{collection.name}'.")

def query_db(collection, model: SentenceTransformer, query: str, top_k: int = 3):
    """ค้นหาใน DB ด้วยข้อความ query"""
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = l2_normalize(q_emb)

    res = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=top_k,
        include=["metadatas", "documents", "distances"]
    )

    print(f"\n=== Query: {query} ===")
    for rank, (meta, doc, dist) in enumerate(zip(res["metadatas"][0], res["documents"][0], res["distances"][0]), start=1):
        print(f"#{rank} -> {meta['filename']} ({meta['label']}) distance={dist:.4f}")
        print("   desc:", doc)

# ---------------- Main ----------------
def main():
    # 1. เตรียม path ของรูป
    all_images = DOG_IMAGES + CAT_IMAGES
    labels = ["dog"] * len(DOG_IMAGES) + ["cat"] * len(CAT_IMAGES)

    # 2. โหลด CLIP model
    model = SentenceTransformer("clip-ViT-B-32")

    # 3. โหลดรูปจาก local
    pil_images = load_images(all_images)

    # 4. แปลงเป็น embedding
    img_embs = encode_images(model, pil_images)

    # 5. ตั้งค่า Chroma
    client = chromadb.Client(chromadb.config.Settings(persist_directory=PERSIST_DIR))
    collection = setup_collection(client, COLLECTION_NAME)

    # 6. เพิ่มรูปเข้า DB
    add_images_to_db(collection, all_images, labels, img_embs)

    # 7. Query หา "หมา"
    query_db(collection, model, "dog", top_k=4)

if __name__ == "__main__":
    main()
