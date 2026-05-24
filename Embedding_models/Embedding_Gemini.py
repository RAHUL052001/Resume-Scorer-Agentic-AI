from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Gemini Developer API
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=32
)

documents = [
    "this is my first agentic prject",
    "My birthday is on 5th of june",
    "India have a biggest diversity"
]

all_vectors = [embeddings.embed_query(doc) for doc in documents]

print(all_vectors)
# Vertex AI
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    project="my-project",
    vertexai=True,
)



