import chromadb
import numpy as np
from chromadb.api.types import EmbeddingFunction

# Define a custom embedding function for face embeddings
class FaceEmbeddingFunction(EmbeddingFunction):
    def __call__(self, docs):
        """
        This method should return a list of embeddings for the given list of input data (docs).
        Each embedding should be a fixed-size numpy array.
        """
        # Convert face images or feature vectors into embeddings (Modify this as per your model)
        return [np.random.rand(512).tolist() for _ in docs]  # Example: Random 512-dim embeddings

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient()  # Or chromadb.Client() for in-memory storage

# Use the custom face embedding function
face_embedding_fn = FaceEmbeddingFunction()

# Collection name
collection_name = "face"

# Check if the collection already exists
if collection_name in [coll.name for coll in chroma_client.list_collections()]:
    face_collection = chroma_client.get_collection(name=collection_name, embedding_function=face_embedding_fn)
else:
    face_collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=face_embedding_fn  # Now correctly typed for face embeddings
    )

# Print confirmation
print(f"Collection '{collection_name}' is ready!")
