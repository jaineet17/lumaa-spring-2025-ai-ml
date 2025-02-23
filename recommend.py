import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Preprocess text and compute TF-IDF vectors
def vectorize_text(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    return tfidf, tfidf_matrix

# Compute similarity and recommend movies
def recommend_movies(query, df, tfidf, tfidf_matrix, top_n=5):
    query_vector = tfidf.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    df['similarity'] = similarity_scores
    recommendations = df.sort_values(by='similarity', ascending=False).head(top_n)
    return recommendations[['title', 'similarity']]

# Main function
def main(query):
    df = load_data('movies.csv')
    tfidf, tfidf_matrix = vectorize_text(df)
    recommendations = recommend_movies(query, df, tfidf, tfidf_matrix)
    print("Top 5 Recommended Movies:")
    for idx, row in recommendations.iterrows():
        print(f"{idx + 1}. {row['title']} (Similarity Score: {row['similarity']:.2f})")

if __name__ == "__main__":
    import sys
    query = sys.argv[1]
    main(query)