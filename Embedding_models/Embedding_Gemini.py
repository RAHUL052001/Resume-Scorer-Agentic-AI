import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# We request 128 (the model's lowest natively supported size) 
# and slice it down to 8 in Python to keep the data perfectly clean.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", 
    output_dimensionality=128
)

documents = ["This is my first document.", "This is my second document.", "This is my third document."]

# Fetch the full embedding
raw_vector = embeddings.embed_query(documents[0])

# Slice to exactly 8 dimensions and round to 2 decimal places
short_vector = [round(num, 2) for num in raw_vector[:8]]

print(f"Vector Length: {len(short_vector)}")
print(f"Clean Short Vector: {short_vector}")