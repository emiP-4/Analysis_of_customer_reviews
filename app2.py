import streamlit as st
import pandas as pd
import numpy as np
import joblib
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

st.set_page_config(page_title="Projet NLP - Avis Clients", layout="wide")

# ==========================================
# 1. CHARGEMENT EN CACHE (POUR LA PERFORMANCE)
# ==========================================
@st.cache_resource
def load_models():
    # Chargement des modèles entraînés
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    model_lr = joblib.load('logistic_model.pkl')
    w2v_model = Word2Vec.load("word2vec.model")
    
    # Pipelines HuggingFace (Génératif & QA)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return tfidf, model_lr, w2v_model, summarizer, qa_model

@st.cache_data
def load_data(_w2v_model):
    # Chargement des données et pré-calcul des vecteurs de phrases pour la recherche
    df = pd.read_csv("app_data.csv")
    df = df.dropna(subset=['clean_final'])
    
    # Fonction de vectorisation (identique à ton notebook)
    def get_sentence_vector(sentence):
        words = str(sentence).split()
        vectors = [_w2v_model.wv[w] for w in words if w in _w2v_model.wv]
        if len(vectors) == 0:
            return np.zeros(_w2v_model.vector_size)
        return np.mean(vectors, axis=0)
    
    df['vector'] = df['clean_final'].apply(get_sentence_vector)
    return df

# Initialisation
with st.spinner("Chargement des modèles d'Intelligence Artificielle..."):
    tfidf, model_lr, w2v_model, summarizer, qa_model = load_models()
    df = load_data(w2v_model)

# ==========================================
# 2. NAVIGATION ET INTERFACE
# ==========================================
st.title("Projet NLP : Analyse et Interaction avec les Avis Clients")
menu = ["Prédiction", "Summary", "Explanation", "Information Retrieval", "QA", "RAG"]
choix = st.sidebar.selectbox("Choisissez une fonctionnalité", menu)

# --- A. PRÉDICTION ---
if choix == "Prédiction":
    st.header("🔮 Prédiction de Sentiment")
    texte_utilisateur = st.text_area("Entrez un avis client :", "The customer service was really great and fast!")
    
    if st.button("Prédire"):
        vecteur = tfidf.transform([texte_utilisateur])
        prediction = model_lr.predict(vecteur)[0]
        
        if prediction == "positive":
            st.success(f"Sentiment : POSITIF 🟢")
        elif prediction == "negative":
            st.error(f"Sentiment : NÉGATIF 🔴")
        else:
            st.warning(f"Sentiment : NEUTRE 🟡")

# --- B. SUMMARY ---
elif choix == "Summary":
    st.header("📝 Résumé Automatique")
    texte_long = st.text_area("Collez un long avis ou un article ici :", height=200)
    
    if st.button("Résumer"):
        if len(texte_long.split()) < 20:
            st.warning("Le texte est trop court pour être résumé.")
        else:
            with st.spinner("Génération du résumé..."):
                resultat = summarizer(texte_long, max_length=50, min_length=10, do_sample=False)
                st.info("**Résumé :** " + resultat[0]['summary_text'])

# --- C. EXPLANATION ---
elif choix == "Explanation":
    st.header("🧠 Explication des décisions du Modèle")
    texte = st.text_area("Entrez un texte pour voir ce qui influence le modèle :", "The price is good but the service is terrible")
    
    if st.button("Expliquer"):
        # Transformation TF-IDF
        vecteur = tfidf.transform([texte])
        feature_names = tfidf.get_feature_names_out()
        
        # Récupération des mots non nuls dans la phrase
        mots_indices = vecteur.nonzero()[1]
        
        explications = {}
        for idx in mots_indices:
            mot = feature_names[idx]
            poids_tfidf = vecteur[0, idx]
            # Le coefficient de la classe 'positive' (index 2 généralement selon le tri de scikit-learn)
            # On prend la moyenne ou l'impact direct pour simplifier
            poids_modele = model_lr.coef_[0][idx] 
            impact = poids_tfidf * poids_modele
            explications[mot] = impact
            
        if explications:
            df_expl = pd.DataFrame.from_dict(explications, orient='index', columns=['Impact (Positif > 0 > Négatif)'])
            st.write("Voici l'impact mathématique de chaque mot sur la prédiction finale :")
            st.bar_chart(df_expl)
        else:
            st.warning("Aucun mot de votre phrase n'est connu du modèle TF-IDF.")

# --- D. INFORMATION RETRIEVAL ---
elif choix == "Information Retrieval":
    st.header("🔍 Recherche Sémantique (Information Retrieval)")
    requete = st.text_input("Que recherchez-vous ? (ex: bad customer service)")
    
    if st.button("Rechercher"):
        # Fonction vectorisation de la requête
        mots = requete.split()
        vecteurs_requete = [w2v_model.wv[w] for w in mots if w in w2v_model.wv]
        
        if len(vecteurs_requete) == 0:
            st.error("Aucun mot de votre requête n'est dans le vocabulaire.")
        else:
            vec_requete = np.mean(vecteurs_requete, axis=0)
            # Calcul des similarités avec tout le dataframe
            df['similarite'] = df['vector'].apply(lambda x: cosine_similarity([vec_requete], [x])[0][0])
            top_results = df.nlargest(5, 'similarite')
            
            st.write("Top 5 des avis les plus pertinents :")
            st.table(top_results[['clean_final', 'similarite']])

# --- E. QA (QUESTION ANSWERING) ---
elif choix == "QA":
    st.header("💬 Question / Réponse sur un texte")
    contexte = st.text_area("Contexte (Avis complet) :", "I signed my insurance contract in 2010. The price was 50 euros per month.")
    question = st.text_input("Question :", "How much was the price per month?")
    
    if st.button("Trouver la réponse"):
        with st.spinner("Recherche de la réponse..."):
            reponse = qa_model(question=question, context=contexte)
            st.success(f"**Réponse extraite :** {reponse['answer']}")
            st.caption(f"Score de confiance : {round(reponse['score']*100, 2)}%")

# --- F. RAG (RETRIEVAL-AUGMENTED GENERATION) ---
elif choix == "RAG":
    st.header("🤖 RAG : Poser une question sur l'ensemble de vos avis")
    st.markdown("*Ce module combine l'Information Retrieval et le Question Answering.*")
    
    question_rag = st.text_input("Posez votre question :", "What are the main problems with the insurance ?")
    
    if st.button("Analyser via RAG"):
        with st.spinner("Étape 1 : Recherche des documents pertinents (Retrieval)..."):
            mots = question_rag.split()
            vecteurs_requete = [w2v_model.wv[w] for w in mots if w in w2v_model.wv]
            
            if len(vecteurs_requete) == 0:
                st.error("Je ne comprends pas assez de mots dans cette question pour chercher.")
            else:
                vec_requete = np.mean(vecteurs_requete, axis=0)
                df['similarite'] = df['vector'].apply(lambda x: cosine_similarity([vec_requete], [x])[0][0])
                
                # On prend les 3 meilleurs avis pour former le contexte
                top_3 = df.nlargest(3, 'similarite')['clean_final'].tolist()
                contexte_global = " . ".join(top_3)
                
        with st.spinner("Étape 2 : Lecture par le LLM et génération de la réponse (Generation)..."):
            reponse = qa_model(question=question_rag, context=contexte_global)
            
            st.success("### Réponse du modèle RAG :")
            st.write(f"**{reponse['answer']}**")
            
            st.info("💡 **Sources utilisées (Contexte) :**")
            for doc in top_3:
                st.write(f"- *{doc}*")