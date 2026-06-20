from vector_embedding_db import collection


def main() -> None:
    data = collection.get(include=["documents", "metadatas"])
    print("Stored ChromaDB data:")
    for idx, item_id in enumerate(data["ids"]):
        document = data["documents"][idx] if data["documents"] else None
        metadata = data["metadatas"][idx] if data["metadatas"] else None
        print(f"ID: {item_id}")
        print(f"Document: {document}")
        print(f"Metadata: {metadata}")
        print("---")


if __name__ == "__main__":
    main()
