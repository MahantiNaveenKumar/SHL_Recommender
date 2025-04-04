import pandas as pd
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample titles
titles = [
    "Verbal Reasoning Test",
    "Numerical Reasoning Test",
    "Situational Judgement Test",
    "Leadership Assessment",
    "Teamwork Evaluation"
]

# Generate embeddings
embeddings = model.encode(titles)

# Save to CSV
df = pd.DataFrame({
    'title': titles,
    'embedding': embeddings.tolist()
})

df.to_csv("data.csv", index=False)
print("✅ data.csv has been created!")
