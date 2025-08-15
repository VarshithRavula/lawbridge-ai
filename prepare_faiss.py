from load_data import load_data

file_path = "C:\\Users\\varsh\\Downloads\\justice_sample.csv"
df = load_data(file_path)

# Prepare cases for FAISS
cases = df[["ID", "name", "facts", "first_party_winner"]].dropna(subset=["facts"]).reset_index(drop=True)

print(f"{len(cases)} cases ready for FAISS indexing")
print(cases.head(3))
