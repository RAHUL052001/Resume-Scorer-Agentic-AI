import argparse
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from vector_embedding_db import save_vector, persist

load_dotenv()


def text_to_embedding(text: str):
    """Convert a single line of text to an embedding vector."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=128,
    )
    return embeddings.embed_query(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a single line of text to an embedding and save it to ChromaDB."
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default="This is a sample text line to embed.",
        help="The text line to convert into an embedding.",
    )
    parser.add_argument(
        "--id",
        "-i",
        type=str,
        default=None,
        help="Optional ID to assign to the saved vector in ChromaDB.",
    )
    args = parser.parse_args()

    vector = text_to_embedding(args.text)
    result = save_vector(vector, metadata={"source_text": args.text}, id=args.id)
    persist()

    print("Text saved to ChromaDB")
    print("Text:", args.text)
    print("Embedding length:", len(vector))
    print("Chroma save result:", result)
