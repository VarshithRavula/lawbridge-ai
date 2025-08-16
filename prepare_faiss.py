
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load cases (assuming you already have 'cases' defined or loaded from a file)
# Example:
# import json
# with open("cases.json", "r") as f:
#     cases = json.load(f)

# Load embedding model (free, runs locally)
file_path = "C:\\Users\\varsh\\Downloads\\justice_sample.csv"
df = pd.read_csv(file_path)


# Prepare cases for FAISS
cases = df[["ID", "name", "facts", "first_party_winner"]].dropna(subset=["facts"]).reset_index(drop=True)


# Loading embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  

# Generate embeddings for all cases
embeddings = np.array([model.encode(fact) for fact in cases['facts']], dtype=np.float32)
print("Embeddings generated with shape:", embeddings.shape)
#print(embeddings[0])

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("FAISS index built with", index.ntotal, "vectors.")

# Save FAISS index and cases for later use
faiss.write_index(index, "cases.index")
with open("cases.pkl", "wb") as f:
    pickle.dump(cases, f)

#Index and cases saved successfully
