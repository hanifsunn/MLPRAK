import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

#Impor dataset
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

#Tabel pivot berdasar rating film
rating_pivot = ratings.pivot_table(values='rating', columns='userId', index='movieId').fillna(0)
print('Shape of this pivot table :', rating_pivot.shape)
print(rating_pivot.head())
print("")


nn_algo = NearestNeighbors(metric='cosine')
nn_algo.fit(rating_pivot)

class Recommender:
    def __init__(self):
        self.hist = [] 
        self.ishist = False 
    #Rekomendasi berdasarkan film yang dimasukkan
    def recommend_on_movie(self, movie, n_recommend=5):
        self.ishist = True
        movieid = int(movies[movies['title'] == movie]['movieId'].iloc[0])
        self.hist.append(movieid)
        distance, neighbors = nn_algo.kneighbors([rating_pivot.loc[movieid]], n_neighbors=n_recommend + 1)
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommendations = [(movies[movies['movieId'] == mid]['title'].iloc[0], 1 - dist) for mid, dist in zip(movieids, distance[0]) if mid != movieid]
        return recommendations[:n_recommend]
    
    #Rekomendasi berdasarkan daftar histori di self.hist
    def recommend_on_history(self, n_recommend=5):
        if not self.ishist:
            print('Tidak ada histori yang ditemukan')
            return []
        history = np.array([list(rating_pivot.loc[mid]) for mid in self.hist])
        distance, neighbors = nn_algo.kneighbors([np.average(history, axis=0)], n_neighbors=n_recommend + len(self.hist))
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommendations = [(movies[movies['movieId'] == mid]['title'].iloc[0], 1 - dist) for mid, dist in zip(movieids, distance[0]) if mid not in self.hist]
        return recommendations[:n_recommend]

recommender = Recommender() 

print("Rekomendasi Berdasarkan Histori")
print(recommender.recommend_on_history())  
print("")

#SUPERMAN (1978)
#Rekomendasi berdasarkan film ini 
recommendations = recommender.recommend_on_movie('Superman (1978)')
print("Rekomendasi Berdasarkan Film Superman (1978)")
for title, a in recommendations:
    print(f"Film: {title}, Skor Similarity: {a:.2f}")
print("")
#Rekomendasi berdasarkan film yang telah ditonton sebelumnya, dan kali ini ada film dalam histori.
print("Rekomendasi Berdasarkan Histori")
histori = recommender.recommend_on_history()
for title, a in histori:
    print(f"Film: {title}, Skor Similarity: {a:.2f}")
print("")



#Runaway Bride (1999)
recommendations = recommender.recommend_on_movie('Runaway Bride (1999)')
print("Rekomendasi Berdasarkan Film Runaway Bride (1999)")
for title, a in recommendations:
    print(f"Film: {title}, Skor Similarity: {a:.2f}")
print("")
print("Rekomendasi Berdasarkan Histori")
histori = recommender.recommend_on_history()
for title, a in histori:
    print(f"Film: {title}, Skor Similarity: {a:.2f}")
print("")
