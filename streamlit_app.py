
import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_resource
def load_data(): 
    
    # Load CSV files
    train_df = pd.read_csv("train.csv")
    anime_df = pd.read_csv("combined_anime.csv")
    tfidf_matrix = pd.read_csv("tfidf_matrix_df.csv")

    return train_df, anime_df, tfidf_matrix

@st.cache_resource
def load_model():

    # Load the SVD model
    with open("knnbaseline_model.pkl", "rb") as f: 
        model = pickle.load(f) 

    return model

# Load all resources
train_df, anime_df, tfidf_matrix = load_data()
model = load_model()


def get_similar_anime(input_anime, anime_df, tfidf_matrix, top_n=10):
    # Verify if the anime exists in the dataset
    if input_anime not in anime_df['name'].values:
        return None

    # Find the index of the input anime in the DataFrame
    input_anime_idx = anime_df.index[anime_df['name'] == input_anime].tolist()[0]

    # Extract the TF-IDF vector for the input anime
    input_vector = tfidf_matrix.iloc[input_anime_idx, :-1].values.reshape(1, -1)

    # Compute cosine similarities between the input anime and all other animes
    similarities = cosine_similarity(input_vector, tfidf_matrix.iloc[:, :-1]).flatten()

    # Identify the indices of the top N most similar animes, excluding the input itself
    similar_anime_indices = similarities.argsort()[-(top_n + 1):][::-1][1:]

    # Return the names of the top N most similar animes
    return anime_df.loc[similar_anime_indices, 'name'].tolist()


def collaborative_filtering(user_id, n=10):
    best_anime = anime_df.sort_values(by='rating',ascending=False).head(10)
    best_anime = best_anime['anime_id'].tolist()
    # Get unique anime IDs
    anime_ids = anime_df['anime_id'].unique()

    # Handle cold start: if user is new, return top popular anime
    if user_id not in train_df['user_id'].unique():
        return anime_df[anime_df['anime_id'].isin(best_anime)]['name'].head(n).tolist()

    # Identify the anime the user has already rated
    rated_anime = train_df[train_df['user_id'] == user_id]['anime_id'].tolist()

    # Generate predictions for unseen anime
    predictions = [
        model.predict(user_id, anime_id) 
        for anime_id in anime_ids if anime_id not in rated_anime
    ]

    # Sort predictions based on the rating and pick top N
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n_anime_ids = [pred.iid for pred in predictions[:n]]

    # Fetch names of the recommended anime
    recommended_anime = anime_df[anime_df['anime_id'].isin(top_n_anime_ids)]['name'].tolist()

    return recommended_anime

def main():
    st.sidebar.title("Menu")
    section = st.sidebar.radio(" ", ["Making Recommendations"])

    if section == "Making Recommendations":
        st.title("Making Recommendations")

        options = ["Content-Based Anime Recommender", "Collaboration Filtering Anime Recommender"]
        part = st.sidebar.selectbox("Recommender Options", options)

        if part == "Content-Based Anime Recommender":
            st.subheader("Search Anime")

            selected_anime = st.selectbox("Select Anime", anime_df['name'].values)

            if st.button("Get Recommended Anime"):
                recommended_anime = get_similar_anime(selected_anime, anime_df, cosine_sim)

                st.write("### Recommended Anime:")
                for i, anime in enumerate(recommended_anime, 1):
                    st.write(f"{i}. {anime}")

        elif part == "Collaboration Filtering Anime Recommender":
            st.subheader("Search User")

            selected_user = st.number_input("Enter User ID:", min_value=1, format="%d")

            if st.button("Get Recommended Anime"):
                recommendations = collaborative_filtering(selected_user)

                st.write("### Recommended Anime:")
                for i, anime in enumerate(recommendations):
                    st.write(f"{i+1}. {anime}")
if __name__ == "__main__":
    main()


            
                        
            

                 
