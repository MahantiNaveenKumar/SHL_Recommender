from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import ast

app = FastAPI()

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the CSV
df = pd.read_csv("data.csv")

# Convert stringified lists to real lists
df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

@app.get("/")
def root():
    return {"message": "Welcome to the SHL Recommender API 🚀"}

@app.get("/recommend")
def recommend(text: str = Query(..., description="Enter a query like 'problem solving'")):
    # Encode user input
    query_embedding = model.encode([text], convert_to_tensor=True)

    recommendations = []
    for _, row in df.iterrows():
        # Ensure embedding is a tensor
        row_embedding = torch.tensor(row['embedding'])
        score = util.pytorch_cos_sim(query_embedding, row_embedding)[0][0].item()
        recommendations.append((row['title'], score))

    # Sort and return top 5
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return {"results": recommendations[:5]}
