# import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import missingno as msno
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM
import warnings
warnings.filterwarnings("ignore")



# import data
df = pd.read_csv("D:/0Tugas/0FILE UNAIR/0FILE MATERI/0SEMESTER 4/Machine Learning/MLPRAK/Latihan_FCM_187221048/apple_quality.csv")
print("Data 5 baris pertama: \n", df.head())

print("\nbentuk data: ", df.shape)

# lihat data
print("\ndataset full : \n", df)

# Cari missing value
missing_values = df.isnull().sum()
print("\nJumlah missing value untuk setiap atribut:\n", missing_values)

# Hapus baris dengan missing value (jika ada)
df.dropna(inplace=True)

# Periksa kembali jumlah baris dan kolom setelah menghapus missing value
print("\nJumlah baris dan kolom setelah menghapus missing value:\n", df.shape)

# lihat data setelah menghapus missing value
print("\nData setelah menghapus missing value:\n", df)

# fungsi untuk mendeteksi outlier menggunakan metode IQR
def detect_outliers(df, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = list(set(outlier_indices))
    return outlier_indices

# Tentukan fitur yang ingin diperiksa untuk outlier
outlier_features = df.columns[:-1]  

# Deteksi outlier
outlier_indices = detect_outliers(df, outlier_features)
print("\nJumlah outlier:", len(outlier_indices))

# Hapus baris dengan outlier (jika ada)
df.drop(outlier_indices, inplace=True)

# Periksa kembali jumlah baris sdan kolom setelah menghapus outlier
print("\nJumlah baris dan kolom setelah menghapus outlier:\n", df.shape)

# Lihat data setelah menghapus outlier
print("\nData setelah menghapus outlier:\n", df)


# amati bentuk visual masing-masing fitur
# Histogram untuk setiap fitur
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
for i, col in enumerate(df.columns[:-1]):
    plt.subplot(2, 4, i + 1)  # Create subplots in a 2x4 grid
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()
# Hitung jumlah masing-masing kualitas
species_counts = df['Quality'].value_counts()
# Buat pie chart
plt.figure(figsize=(8, 6))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Quality')
plt.axis('equal')  # Memastikan lingkaran berbentuk lingkaran
plt.show()

# Membangun FCM
nmpy = df.iloc[:, :-1].values  # Memilih fitur-fitur numerik sebagai input
model = FCM(n_clusters=3)
model.fit(nmpy)

# Mendapatkan cluster centers dan labels
centers = model.centers
labels = model.predict(nmpy)

# Memvisualisasikan hasil clustering
plt.scatter(nmpy[labels == 0, 2], nmpy[labels == 0, 3], s=10, c='r', label='Cluster 1')
plt.scatter(nmpy[labels == 1, 2], nmpy[labels == 1, 3], s=10, c='b', label='Cluster 2')
plt.scatter(nmpy[labels == 2, 2], nmpy[labels == 2, 3], s=10, c='g', label='Cluster 3')
plt.scatter(centers[:, 2], centers[:, 3], s=300, c='black', marker='+', label='Centroids')
plt.title('Clustering')
plt.legend()
plt.show()

# Melihat nilai silhouette score
score2 = silhouette_score(nmpy, labels)
print("Silhouette Score: ", score2)