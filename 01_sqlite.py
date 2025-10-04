import sqlite3
import sqlite_vec
import hashlib
import json
from typing import List, Tuple

# ---------- demo embedding ----------
def embed_text(text: str, dim: int = 8) -> List[float]:
    """Simple deterministic embedding for demo purposes.
    Uses SHA256 of the text to produce a repeatable vector in range [-1, 1].
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec = []
    for i in range(dim):
        b1 = h[(i * 2) % len(h)]
        b2 = h[(i * 2 + 1) % len(h)]
        val = (b1 << 8) | b2
        f = (val / 65535.0) * 2.0 - 1.0
        vec.append(f)
    return vec

# ---------- KNN Search ----------
def knn_search(conn, query_text: str, k: int = 2) -> list[Tuple[int, float, str, list[float]]]:
    qvec = embed_text(query_text, dim=8)
    qjson = json.dumps(qvec)

    sql = """
    SELECT
        v.rowid       AS id,
        v.distance    AS distance,
        d.content     AS content
    FROM vec_docs AS v
    JOIN docs AS d ON d.id = v.rowid
    WHERE v.embedding MATCH ?
      AND k = ?              -- ⭐ บอกจำนวนผลลัพธ์ที่ต้องการ
    ORDER BY v.distance;
    """
    cur = conn.cursor()
    res = cur.execute(sql, (qjson, k)).fetchall()

    output = []
    for _id, dist, content in res:
        output.append((_id, float(dist), content, []))
    return output

# ---------- ตัวอย่างใช้งาน ----------
if __name__ == "__main__":
    # ---------- SQLite ----------
    conn = sqlite3.connect("demo.db")  # หรือ ":memory:" for ephemeral
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.enable_load_extension(True)

    # ⭐ โหลด extension ผ่าน sqlite-vec (ไม่ต้องหาพาธเอง)
    sqlite_vec.load(conn)
    # หรือถ้าอยาก explicit: conn.load_extension(sqlite_vec.loadable_path())

    cur = conn.cursor()

    # ---------- ตารางข้อความ ----------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY,
        content TEXT NOT NULL
    );
    """)

    # Insert docs only if table is empty (idempotent)
    cur.execute("SELECT COUNT(*) FROM docs;")
    # ---------- ข้อมูลเอกสาร ----------
    docs = [
        (1, "The quick brown fox jumps over the lazy dog"),
        (2, "A fast auburn fox leaps above a sleepy canine"),
        (3, "An article about database systems and vector search"),
        (4, "Deep learning and embeddings for natural language processing"),
    ]
    docs_count = cur.fetchone()[0]
    if docs_count == 0:
        cur.executemany("INSERT INTO docs(id, content) VALUES (?, ?);", docs)

    # ---------- ตารางเวกเตอร์ ----------
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_docs USING vec0(
        embedding FLOAT[8]
    );
    """)

    # ใส่ embeddings เป็น JSON string แต่เฉพาะเมื่อไม่มีข้อมูลอยู่แล้ว
    cur.execute("SELECT COUNT(*) FROM vec_docs;")
    vec_count = cur.fetchone()[0]
    if vec_count == 0:
        rows = []
        for _id, text in docs:
            emb = embed_text(text, dim=8)
            rows.append((_id, json.dumps(emb)))

        cur.executemany("INSERT INTO vec_docs(rowid, embedding) VALUES (?, ?);", rows)
        conn.commit()

    # ---------- KNN Search ----------
    query = "fox dog"
    results = knn_search(conn, query, k=2)

    print(f"Query: {query!r}")
    for _id, dist, content, _ in results:
        print(f"- id={_id}  distance={dist:.6f}  content={content}")

    # Query: สร้างเวกเตอร์จากข้อความด้วย embed_text แล้วส่งเวกเตอร์นี้ไปค้นหา
    text_query = query
    query_vec = embed_text(text_query, dim=8)
    query_vec_json = json.dumps(query_vec)
    res = cur.execute(
        """
        SELECT rowid, distance
        FROM vec_docs
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT 2;
        """,
        (query_vec_json,)
    ).fetchall()

    print("\nKNN embed_text:")
    for rowid, distance in res:
        print(f"- rowid={rowid}  distance={float(distance):.12f}")
