import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="backend/chromadb")

# List all collection names
collections = chroma_client.list_collections()
print("Available Collections:")
for coll in collections:
    print(coll)  # Print the collection name directly

# Load the collection
collection = chroma_client.get_collection(name="face_embeddings")  # Use your actual collection name

# Retrieve stored embeddings
stored_data = collection.get()  # Fetch all data

# Print the entire stored data to understand its structure
print("Stored Data:", stored_data)

# Ensure that 'ids' and 'metadatas' exist in the stored data
ids = stored_data.get("ids", [])
metadatas = stored_data.get("metadatas", [])

# Print stored embeddings and metadata
if ids and metadatas:
    print("Stored Embeddings:")
    for id, metadata in zip(ids, metadatas):
        print(f"ID: {id}, Metadata: {metadata}")
else:
    print("No data found in the collection.")
