import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load datasets
movies = pd.read_csv('D:/0Tugas/0FILE UNAIR/0FILE MATERI/0SEMESTER 4/Machine Learning/MLPRAK/UAS/movies.csv')
print("Dataset Movies")
print('Shape of this dataset :', movies.shape)
print(movies.head())
print("")

ratings = pd.read_csv('D:/0Tugas/0FILE UNAIR/0FILE MATERI/0SEMESTER 4/Machine Learning/MLPRAK/ratings.csv/ratings.csv')
print("Dataset Ratings")
print('Shape of this dataset :', ratings.shape)
print(ratings.head())
print("")

users = pd.read_csv('D:/0Tugas/0FILE UNAIR/0FILE MATERI/0SEMESTER 4/Machine Learning/MLPRAK/UAS/users.csv')
print("Dataset Users")
print('Shape of this dataset :', users.shape)
print(users.head())
print("")

# Pivot table
rating_pivot = ratings.pivot_table(values='rating', columns='userId', index='movieId').fillna(0)
print('Shape of this pivot table :', rating_pivot.shape)
print(rating_pivot.head())
print("")

# Train the Nearest Neighbors algorithm
nn_algo = NearestNeighbors(metric='cosine')
nn_algo.fit(rating_pivot)

class Recommender:
    def __init__(self):
        self.hist = [] 
        self.ishist = False 
        
    def recommend_on_movie(self, movie, n_recommend=5):
        self.ishist = True
        movieid = movies[movies['title'] == movie]['movieId']
        if movieid.empty:
            return None
        self.hist.append(movieid.iloc[0])
        distance, neighbors = nn_algo.kneighbors([rating_pivot.loc[movieid.iloc[0]]], n_neighbors=n_recommend + 1)
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommendations = [(movies[movies['movieId'] == mid]['title'].iloc[0], 1 - dist) for mid, dist in zip(movieids, distance[0]) if mid != movieid.iloc[0]]
        return recommendations[:n_recommend]
    
    def recommend_on_history(self, n_recommend=5):
        if not self.ishist:
            print('No history found')
            return []
        history = np.array([list(rating_pivot.loc[mid]) for mid in self.hist])
        distance, neighbors = nn_algo.kneighbors([np.average(history, axis=0)], n_neighbors=n_recommend + len(self.hist))
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommendations = [(movies[movies['movieId'] == mid]['title'].iloc[0], 1 - dist) for mid, dist in zip(movieids, distance[0]) if mid not in self.hist]
        return recommendations[:n_recommend]

recommender = Recommender() 

# Input loop
while True:
    movie_title = input("Masukkan Judul Film: ")
    if movie_title.lower() == 'exit':
        break
    
    recommendations = recommender.recommend_on_movie(movie_title)
    while recommendations is None:
        print(f"Film '{movie_title}' tidak ditemukan. Coba lagi.")
        movie_title = input("Masukkan Judul Film (atau ketik 'exit' lalu Enter untuk berhenti): ")
        if movie_title.lower() == 'exit':
            break
        recommendations = recommender.recommend_on_movie(movie_title)
    
    if movie_title.lower() == 'exit':
        break
    
    print(f"Rekomendasi Berdasarkan Film '{movie_title}':")
    for title, score in recommendations:
        print(f"Movie: {title}, Similarity Score: {score:.2f}")
    print("")

    print("Rekomendasi Berdasarkan Histori Pencarian:")
    history_recommendations = recommender.recommend_on_history()
    for title, score in history_recommendations:
        print(f"Movie: {title}, Similarity Score: {score:.2f}")
    print("")

print("Program selesai.")
