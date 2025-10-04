from typing import List, Tuple
import chromadb
from chromadb.utils import embedding_functions

def make_docs() -> List[Tuple[str, str]]:
	# (id, content)
	return [
		("1", "The quick brown fox jumps over the lazy dog"),
		("2", "A fast auburn fox leaps above a sleepy canine"),
		("3", "An article about database systems and vector search"),
		("4", "Deep learning and embeddings for natural language processing"),
	]


def main() -> None:
	# Choose an embedding function from chromadb utils. This uses
	# sentence-transformers under the hood (no external API keys required).
	embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
		model_name="all-MiniLM-L6-v2",
	)

	# Create a local in-memory client. If you want persistence, pass
	# chromadb.config.Settings(..., persist_directory="./chroma_db") to Client().
	client = chromadb.Client()

	# create_collection will return existing collection if name already exists
	collection = client.get_or_create_collection(
		name="demo_collection",
		embedding_function=embedding_fn,
	)

	docs = make_docs()
	ids = [d[0] for d in docs]
	documents = [d[1] for d in docs]

	# Add documents. If documents with same ids already exist Chroma will raise,
	# so here we try to delete any existing ids first for idempotence in examples.
	try:
		collection.delete(ids=ids)
	except Exception:
		# ignore if they don't exist
		pass

	collection.add(ids=ids, documents=documents)

	# Query: similar to the sqlite example
	query = "fox dog"
	result = collection.query(
		query_texts=[query],
		n_results=2,
		include=["distances", "documents", "metadatas"],
	)

	print(f"Query: {query!r}")
	# result is a dict-of-lists; extract the first query result
	docs_out = result.get("documents", [[]])[0]
	dists_out = result.get("distances", [[]])[0]
	for doc, dist in zip(docs_out, dists_out):
		print(f"- distance={dist:.6f}  content={doc}")


if __name__ == "__main__":
	main()

