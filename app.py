import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load assessment data with embeddings
with open("shl_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

# App title
st.title("🔍 SHL Assessment Recommendation Engine")

# Text input
user_input = st.text_area("Paste Job Description Here", height=200)

# Button click
if st.button("Recommend Assessments"):
    if user_input.strip():
        input_embedding = model.encode([user_input])
        assessment_embeddings = np.vstack(df["embedding"].values)
        similarities = cosine_similarity(input_embedding, assessment_embeddings)[0]
        top_indices = similarities.argsort()[::-1][:3]
        results = df.iloc[top_indices][["name", "description", "url", "test_type", "duration"]]

        for idx, row in results.iterrows():
            st.subheader(f"✅ {row['name']}")
            st.write(row['description'])
            st.markdown(f"[View Assessment]({row['url']})")
            st.write(f"**Test Type:** {row['test_type']} | **Duration:** {row['duration']}")
    else:
        st.warning("Please enter a job description.")
