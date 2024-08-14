
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Import Data
data = pd.read_excel("D:/0Tugas/0FILE UNAIR/0FILE MATERI/0SEMESTER 4/Machine Learning/MLPRAK/UTS/diabetes.xlsx")

# Copy data for preprocessing
df = data.copy()

# Menghitung Jumlah missing data
mvtotal = df.isnull().sum()
print("Jumlah data yang missing")
print(mvtotal)
print('\n')

# Imputasi missing value dengan median untuk setiap kolom dalam daftar
df.fillna(df.median(), inplace=True)

# Menghitung ulang Jumlah missing data
mvtotal = df.isnull().sum()
print("Jumlah data yang missing setelah preprocessing")
print(mvtotal)
print('\n')

# Deteksi outlier
def deteksi_outlier(dataframe):
    outliers = {}
    for column in dataframe.columns:
        threshold = 4.5
        median = np.median(dataframe[column])
        median_absolute_deviation = np.median(np.abs(dataframe[column] - median))
        
        column_outliers = []
        for y in dataframe[column]:
            median_difference = (y - median) / median_absolute_deviation
            if np.abs(median_difference) > threshold:
                column_outliers.append(y)
        outliers[column] = column_outliers
    return outliers

outliers = deteksi_outlier(df)
print('\nOUTLIER sebelum diimputasi : ', outliers)
print('\n')

# Mengganti outlier dengan nilai median
def ganti_outlier_dengan_median(dataframe):
    outliers = {}
    for column in dataframe.columns:
        threshold = 4.5
        median = np.median(dataframe[column])
        median_absolute_deviation = np.median(np.abs(dataframe[column] - median))
        
        column_outliers = []
        for i, y in enumerate(dataframe[column]):
            median_difference = (y - median) / median_absolute_deviation
            if np.abs(median_difference) > threshold:
                dataframe.at[i, column] = median if median > 0 else 1  # Ganti dengan median jika positif, jika tidak, ganti dengan 0
                column_outliers.append(y)
        outliers[column] = column_outliers
    return outliers

outliers = ganti_outlier_dengan_median(df)
print('OUTLIER setelah diimputasi : ', outliers)


# Normalisasi dengan z-score
df_zscore = np.abs(df.apply(lambda x: (x - x.mean()) / x.std()))

# Seleksi fitur dengan chi-square
# Memisahkan fitur dan target
x = df_zscore.iloc[:, :-1]  # Memilih semua kolom kecuali kolom terakhir sebagai fitur
y = df_zscore.iloc[:, -1]   # Memilih kolom terakhir sebagai target

# Menggunakan LabelEncoder untuk mengubah target menjadi label kategori
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Penggunaan SelectKBest untuk mengambil 2 fitur terbaik
best_features = SelectKBest(score_func=chi2, k=2)
x_new = best_features.fit_transform(x, y)

# Menampilkan fitur yang terpilih
print("SELEKSI FITUR CHI SQUARE")
print(x_new.shape)
print(x_new)
