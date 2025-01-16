import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# Collection name
collection_name = "face"

# Check if the collection already exists
if collection_name in [coll.name for coll in chroma_client.list_collections()]:
    face_collection = chroma_client.get_collection(name=collection_name)
else:
    face_collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
