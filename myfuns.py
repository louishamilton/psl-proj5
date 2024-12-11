import pandas as pd
import requests
import pickle
import os
import numpy as np

# Get S matrix from Google Drive (first 100 columns were stored)
file_id = "1LaxOeX0Im8yprIbYbUD8fvPoKkAgkp-_"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(url)
S = pickle.loads(response.content)

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

# Extract movie IDs from S.columns (e.g., "m127" -> 127)
s_movie_ids = [int(col[1:]) for col in S.columns]

# Filter movies DataFrame to include only rows with `movie_id` in `s_movie_ids`
filtered_movies = movies[movies['movie_id'].isin(s_movie_ids)]

# Get ratings into df
url = "https://liangfgithub.github.io/MovieData/ratings.dat?raw=true"
columns = ["UserID", "MovieID", "Rating", "Timestamp"]
ratings_df = pd.read_csv(url, sep="::", names=columns, engine="python")
ratings_df = ratings_df[ratings_df['MovieID'].isin(s_movie_ids)]

# todo filter ratings by movieID

def create_ratings_matrix(ratings_df):
    ratings_matrix = ratings_df.pivot(index="UserID", columns="MovieID", values="Rating")
    ratings_matrix.columns = [f"m{col}" for col in ratings_matrix.columns]  # Rename columns to 'm' + MovieID
    ratings_matrix.index = [f"u{idx}" for idx in ratings_matrix.index]      # Rename rows to 'u' + UserID
    return ratings_matrix

def save_popularity_ranking(rating_matrix, filename="popularity_ranking.csv"):
    """
    Save the ranking of all movies based on their popularity (average rating and review count).

    Args:
        rating_matrix (pd.DataFrame): The ratings matrix.
        filename (str): The file name to save the ranking to.
    """
    movie_avg_ratings = rating_matrix.mean(axis=0, skipna=True)
    movie_review_counts = rating_matrix.notna().sum(axis=0)
    
    # Combine average ratings and review counts into a DataFrame
    popularity_df = pd.DataFrame({
        "AverageRating": movie_avg_ratings,
        "ReviewCount": movie_review_counts
    })
    
    # Filter movies with at least ten reviews
    popularity_df = popularity_df[popularity_df["ReviewCount"] >= 10]

    # Only movies in our subset of 100
    popularity_df = popularity_df[popularity_df.index.isin(S.columns)]
    
    # Sort by average rating, tie break with review count
    popularity_df = popularity_df.sort_values(
        by=["AverageRating", "ReviewCount"], ascending=[False, False]
    )
    
    # Save to file
    popularity_df.to_csv(filename, index=True)

def load_popularity_ranking(filename="popularity_ranking.csv"):
    return pd.read_csv(filename, index_col=0)

# def get_popular_movies(popularity_ranking, top_k=10):
#     top_movies = popularity_ranking.head(top_k)
#     return top_movies

rating_matrix = create_ratings_matrix(ratings_df)

# Call save_popularity_ranking to save the movie ranking if not already stored
ranking_file = "popularity_ranking.csv"
if not os.path.exists(ranking_file):
    save_popularity_ranking(rating_matrix, ranking_file)

# Load Popularity Ranking
popularity_ranking = load_popularity_ranking(ranking_file)

def myIBCF(newuser, similarity_matrix, popularity_ranking, top_k=10):
    """
    Generate movie recommendations for a new user with fallback to popular movies.

    Args:
        newuser (pd.Series): A 3706-by-1 vector of user ratings, indexed by movie IDs (e.g., "m1", "m2", ...).
        similarity_matrix (pd.DataFrame): The filtered similarity matrix.
        popularity_ranking (pd.DataFrame): DataFrame containing movie popularity ranking.
        top_k (int): Number of recommendations to generate (default is 10).

    Returns:
        list: Top-k recommended movies, in the form of column names from the rating matrix (e.g., "m1").
    """
    # Initialize a dictionary to store predictions
    predictions = {}

    # Loop over all movies
    for movie in similarity_matrix.columns:
        if pd.notna(newuser[movie]):  # Skip if the movie is already rated
            continue

        # Get similarity scores for the current movie
        similarities = similarity_matrix.loc[movie]

        # Filter out movies that are not rated by the new user
        rated_movies = similarities.index[similarities.index.map(lambda x: pd.notna(newuser[x]))]
        valid_similarities = similarities[rated_movies]
        valid_ratings = newuser[rated_movies]

        # Calculate the prediction for this movie
        denominator = valid_similarities.sum()
        if denominator > 0:  # Avoid division by zero
            prediction = (valid_similarities * valid_ratings).sum() / denominator
            predictions[movie] = prediction

    # Sort predictions in descending order
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    non_na_recommendations = [movie for movie, pred in sorted_predictions if not pd.isna(pred)]

    # Fallback mechanism for popular movies
    if len(non_na_recommendations) < top_k:
        # Movies already rated by the user
        rated_movies = set(newuser.dropna().index)

        # Movies already in recommendations
        recommended_movies = set(non_na_recommendations)

        # Movies in filtered_movies only

        # Exclude these movies from the fallback list
        excluded_movies = rated_movies.union(recommended_movies)
        remaining_movies = [movie for movie in popularity_ranking.index if movie not in excluded_movies]

        # Add movies from the popularity ranking to fill up the top_k list
        additional_recommendations = remaining_movies[: top_k - len(non_na_recommendations)]
        non_na_recommendations.extend(additional_recommendations)

    return non_na_recommendations[:top_k]

def get_displayed_movies():
    return filtered_movies

def get_recommended_movies(new_user_ratings):
    newuser = pd.Series(data=np.nan, index=S.columns)
    for key in new_user_ratings:
        newuser["m" + str(key)] = new_user_ratings[key]

    results = myIBCF(newuser, S, popularity_ranking, 10)

    # Extract movie IDs from results (e.g., "m127" -> 127)
    m_results = [int(r[1:]) for r in results]

    # Filter the filtered_movies DataFrame to include only rows in m_results
    result = filtered_movies[filtered_movies['movie_id'].isin(m_results)]

    # Ensure the result DataFrame is in the same order as m_results
    result['movie_id'] = pd.Categorical(result['movie_id'], categories=m_results, ordered=True)
    result = result.sort_values('movie_id')
    return result
    #return myIBCF(, S, popularity_ranking, 10)

def get_popular_movies():
    return filtered_movies.head(10)
