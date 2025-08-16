import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

index = faiss.read_index("cases.index")

with open("cases.pkl", "rb") as f:
    cases = pickle.load(f)


model = SentenceTransformer("all-MiniLM-L6-v2")

def search_cases(query, k):
    """
    Search for the top k most similar cases to the given query.
    
    :param query: str, the fact description you want to search for
    :param k: int, number of top similar cases to return
    :return: list of dicts containing case info
    """


    query_vector = np.array([model.encode(query)], dtype = np.float32)
    
    distances, indices = index.search(query_vector, k)

    results = []

    for i, idx in enumerate(indices[0]):
            case_info = {
                "rank": i + 1,
                "ID": cases.loc[idx, "ID"],
                "Name": cases.loc[idx, "name"],
                "Facts": cases.loc[idx, "facts"][:300] + "...",
                "Winner": cases.loc[idx, "first_party_winner"],
                "Distance": distances[0][i]
            }
            results.append(case_info)

    return results

# Example usage
query = "Tenant refuses to pay rent due to maintenance issues"
top_cases = search_cases(query, k=5)

for case in top_cases:
    print(f"\nRank {case['rank']}: {case['Name']} (ID: {case['ID']})")
    print("Facts:", case['Facts'])
    print("Winner:", case['Winner'])
    print("Distance:", case['Distance'])
