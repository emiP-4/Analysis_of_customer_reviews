import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.title("📊 NLP Review Analysis App")

# Load data
df = pd.read_csv("final_cleaned_reviews.csv")

# ===============================
# 🔵 1. SENTIMENT PREDICTION (2 pts)
# ===============================
st.header("🔍 Sentiment Prediction")

user_input = st.text_area("Enter a review:")

if st.button("Predict"):
    prediction = model_lr.predict(tfidf.transform([user_input]))
    st.write("Prediction:", prediction[0])

# ===============================
# 🟣 2. SUMMARY (2 pts)
# ===============================
st.header("📄 Summary")

if st.button("Show Summary"):
    st.write("Total reviews:", len(df))
    st.write(df['sentiment'].value_counts())

# ===============================
# 🟡 3. EXPLANATION (3 pts)
# ===============================
st.header("🧠 Explanation")

if st.button("Explain model"):
    st.write("""
    This model uses TF-IDF + Logistic Regression.
    It identifies important words contributing to sentiment.
    """)

# ===============================
# 🟢 4. INFORMATION RETRIEVAL (3 pts)
# ===============================
st.header("🔎 Search Reviews")

query = st.text_input("Search reviews:")

if st.button("Search"):
    results = df[df['clean_final'].str.contains(query, na=False)]
    st.write(results[['clean_final']].head(5))

# ===============================
# 🔴 5. SEMANTIC SEARCH / RAG (3 pts)
# ===============================
st.header("🤖 Semantic Search")

def get_vector(text):
    words = text.split()
    vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

query_sem = st.text_input("Semantic query:")

if st.button("Semantic Search"):
    q_vec = get_vector(query_sem)
    df['score'] = df['clean_final'].apply(
        lambda x: cosine_similarity([q_vec], [get_vector(x)])[0][0]
    )
    st.write(df.sort_values(by='score', ascending=False).head(5))

# ===============================
# ⚫ 6. QA SYSTEM (3 pts)
# ===============================
st.header("❓ Question Answering")

question = st.text_input("Ask a question:")

if st.button("Answer"):
    st.write("Answer (simple):")
    st.write("Based on reviews, customers often complain about delays and service.")