import numpy as np
import pandas as pd

# Import dataset
movies = pd.read_csv('D:/0Tugas/0FILE UNAIR/0FILE MATERI/0SEMESTER 4/Machine Learning/MLPRAK/UAS/movies.csv')
print("Dataset Movies")
print('Shape of this dataset :',movies.shape)
print(movies.head())
print("")

ratings = pd.read_csv('D:/0Tugas/0FILE UNAIR/0FILE MATERI/0SEMESTER 4/Machine Learning/MLPRAK/ratings.csv/ratings.csv')
print("Dataset Ratings")
print('Shape of this dataset :',ratings.shape)
print(ratings.head())
print("")

users = pd.read_csv('D:/0Tugas/0FILE UNAIR/0FILE MATERI/0SEMESTER 4/Machine Learning/MLPRAK/UAS/users.csv')
print("Dataset Users")
print('Shape of this dataset :',users.shape)
print(users.head())
print("")

rating_pivot = ratings.pivot_table(values='rating',columns='userId',index='movieId').fillna(0)
print('Shape of this pivot table :',rating_pivot.shape)
print(rating_pivot.head())
print("")

from sklearn.neighbors import NearestNeighbors
nn_algo = NearestNeighbors(metric='cosine')
nn_algo.fit(rating_pivot)

class Recommender:
    def __init__(self):
        # This list will stored movies that called atleast ones using recommend_on_movie method
        self.hist = [] 
        self.ishist = False # Check if history is empty
    
    # This method will recommend movies based on a movie that passed as the parameter
    def recommend_on_movie(self,movie,n_reccomend = 5):
        self.ishist = True
        movieid = int(movies[movies['title']==movie]['movieId'].iloc[0])
        self.hist.append(movieid)
        distance,neighbors = nn_algo.kneighbors([rating_pivot.loc[movieid]],n_neighbors=n_reccomend+1)
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommeds = [str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in [movieid]]
        return recommeds[:n_reccomend]
    
    # This method will recommend movies based on history stored in self.hist list
    def recommend_on_history(self,n_reccomend = 5):
        if self.ishist == False:
            return print('No history found')
        history = np.array([list(rating_pivot.loc[mid]) for mid in self.hist])
        distance,neighbors = nn_algo.kneighbors([np.average(history,axis=0)],n_neighbors=n_reccomend + len(self.hist))
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommeds = [str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in self.hist]
        return recommeds[:n_reccomend]

# linitializing the Recommender Object
recommender = Recommender() 

# Recommendation based on past watched movies, but the object just initialized. So, therefore no history found
print("Rekomendasi Berdasarkan Histori")
recommender.recommend_on_history()  
print("")

# Recommendation based on this movie 
print("Rekomendasi Berdasarkan Film Father of the Bride Part II (1995)")
recommender.recommend_on_movie('Father of the Bride Part II (1995)')
print(recommender.recommend_on_movie('Father of the Bride Part II (1995)'))
print("")

# Recommendation based on past watched movies, and this time a movie is there in the history.
print("Rekomendasi Berdasarkan Histori")
recommender.recommend_on_history()
print(recommender.recommend_on_history())
