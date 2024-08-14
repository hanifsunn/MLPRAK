#1 import library
from IPython.display import display
import pandas as pd
from scipy import sparse

#2 import dataset
ratings = pd.read_csv('D:/0Tugas/0FILE UNAIR/0FILE MATERI/0SEMESTER 4/Machine Learning/MLPRAK/Latihan Sistem Rekomendasi_187221048/ratings.csv')
movies = pd.read_csv('D:/0Tugas/0FILE UNAIR/0FILE MATERI/0SEMESTER 4/Machine Learning/MLPRAK/Latihan Sistem Rekomendasi_187221048/movies.csv')

#3 baca dataset ratingss
print("Data Ratings")
print(ratings)

#4 cek null data, hapus apabila ada data yang kosong
missing_values = ratings.isnull().sum()
print("\nJumlah missing value untuk setiap atribut:\n", missing_values)

#5 baca dataset movies
print("\nData Movies")
print(movies)

#6 cek null data, hapus apabila ada data yang kosong
missing_values = movies.isnull().sum()
print("\nJumlah missing value untuk setiap atribut:\n", missing_values)

#7
print("7")
print(ratings.shape)
# jelaskan maksud statement ini, dan jelaskan hasilnya

#8
print("8")
print(movies.shape)
# jelaskan maksud statement ini, dan jelaskan hasilnya


#9 sebutkan atribut-atribut yang sama antara dataset ratings dan movies
# Menampilkan nama kolom dari dataset ratings
print("\nKolom dataset ratings:")
print(ratings.columns)

# Menampilkan nama kolom dari dataset movies
print("\nKolom dataset movies:")
print(movies.columns)
print("")


#10 melakukan merge dataset ratings dan movies
ratings = pd.merge(movies,ratings).drop(['genres','timestamp'],axis=1)
#jelaskan maksud statement ini

print(ratings.shape)
print("")

userRatings = ratings.pivot_table(index=['userId'],columns=['title'],values='rating')
#jelaskan maksud statement ini
print("Rating Film")
print(userRatings)
print("")
#jelaskan hasil dari merge dua dataset tersebut

#11 amati bentuk data
print("Before: ",userRatings.shape)
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0,axis=1)
print("After: ",userRatings.shape)
print("")
#jelaskan maksud statement ini

#12 menghitung matriks korelasi
print("Matriks Korelasi")
corrMatrix = userRatings.corr(method='pearson')
print(corrMatrix)
print("")
#13 romantic_lover movie list
romantic_lover = [
    ("(500) Days of Summer (2009)", 5),
    ("Alice in Wonderland (2010)", 3),
    ("Aliens (1986)", 1),
    ("2001: A Space Odyssey (1968)", 2)
]
#14 find similar movie with romantic lover list
def get_similar(movie_name, rating):
    similar_ratings = corrMatrix[movie_name] * (rating)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    return similar_ratings

similar_movies = pd.concat([get_similar(movie, rating) for movie, rating in romantic_lover], axis=1)
display(similar_movies.head(10))
#jelaskan blok program ini dan hasilnya

#15 comedy movie list
comedy = [
    ("Grumpier Old Men (1995)", 5),
    ("Toy Story (1995)", 3),
    ("Sabrina (1995)", 1),
    ("Four Rooms (1995)", 2)
]
#16 find similar movie with comedy list
similar_movies = pd.concat([get_similar(movie, rating) for movie, rating in comedy], axis=1)
display(similar_movies.head(10))
#jelaskan blok program ini dan hasilnya
