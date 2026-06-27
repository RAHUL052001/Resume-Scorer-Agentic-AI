import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", 
    output_dimensionality=128 # The lowest native size for this model
)

# 1. Three sample documents (Two are about Python, one is about food)
doc1 = "I love writing backend APIs with Python and Django."
doc2 = "Developing web applications using Python is my favorite hobby."
doc3 = "The chef made a delicious pepperoni pizza for dinner."

# 2. Get the full embeddings
v1 = [embeddings.embed_query(doc1)]
v2 = [embeddings.embed_query(doc2)]
v3 = [embeddings.embed_query(doc3)]

print(v1)
print(v2)
print(v3)
# 3. Calculate similarity (Returns a score between 0.0 and 1.0)
similarity_tech = cosine_similarity(v1, v2)[0][0]
similarity_pizza = cosine_similarity(v1, v3)[0][0]

# 4. Print results and slice the vector for clean terminal viewing
print(f"Clean Slice of Doc 1 Vector: {[round(x, 2) for x in v1[0][:6]]}...\n")
print(f"Similarity between Tech & Tech:  {similarity_tech:.4f} (High match!)")
print(f"Similarity between Tech & Pizza: {similarity_pizza:.4f} (Low match!)")

