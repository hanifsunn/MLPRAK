
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
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

# Seleksi fitur dengan PCA
# Memisahkan fitur dan target
x = df_zscore.iloc[:, :-1]  # Memilih semua kolom kecuali kolom terakhir sebagai fitur
y = df_zscore.iloc[:, -1]   # Memilih kolom terakhir sebagai target

# Menggunakan LabelEncoder untuk mengubah target menjadi label kategori
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Apply PCA
pca = PCA(n_components=2) # Menentukan jumlah komponen yang diinginkan
x_pca = pca.fit_transform(x)
print (x_pca)

#Plot hasil PCA
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of diabetes Dataset')
plt.colorbar(label='Diabetes')
plt.show()
