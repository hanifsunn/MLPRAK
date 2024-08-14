#%%
# import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score

# import data
df = pd.read_csv('Mall_Customers.csv')
df.head()

# amati bentuk data
print("BENTUK DATA (baris,kolom)")
print(df.shape)

# Melihat ringkasan statistik deskriptif dari DataFrame 
print("statistik deskriptif dari DataFrame")
print(df.describe())
#tambahkan code disini

# cek null data
print("CEK MISSING VALUE")
print(df.isnull().sum())
 #tambahkan code disini

# cek outlier
print("CEK OUTLIER")
# 6. Cek outlier
print("\nOutlier data : ")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
print(((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).sum())
#Menghilangkan outlier
df = df[~((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
#Periksa outlier kembali
print("\nOutlier data sekarang : ")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
print(((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).sum())

 #tambahkan code disini

#%%
# amati bentuk visual masing-masing fitur
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (15 , 6))
n = 0
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
  n += 1
  plt.subplot(1 , 3 , n)
  plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
  sns.histplot(
    df[x], kde=True,
    stat="density", kde_kws=dict(cut=3), bins = 20)
  plt.title('Distplot of {}'.format(x))
  
 #tambahkan code disini untuk menampilkan figure
plt.show()
 
#%%
# Ploting untuk mencari relasi antara Age , Annual Income and Spending Score
plt.figure(1 , figsize = (15 , 20))
n = 0
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
  for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(3 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.regplot(x = x , y = y , data = df)
    plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()
 #tambahkan code disini untuk menampilkan figure

#%%
# Melihat sebaran Spending Score dan Annual Income pada Gender
plt.figure(1 , figsize = (15 , 8))
for gender in ['Male' , 'Female']:
  plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,
  data = df[df['Gender'] == gender] ,s = 200 , alpha = 0.5 ,
  label = gender)
  plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)')
  plt.title('Annual Income vs Spending Score')
  plt.legend()
plt.show()
 #tambahkan code disini untuk menampilkan figure


#%%
# Merancang K-Means untuk spending score vs annual income
# Menentukan nilai k yang sesuai dengan Elbow-Method
X1 = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
  algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10,max_iter=300, random_state= 111) )
  algorithm.fit(X1)
  inertia.append(algorithm.inertia_)

# Plot bentuk visual elbow
plt.figure(1 , figsize = (15 ,6))
plt.plot(range(1 , 11) , inertia , 'o')
plt.plot(range(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()
 #tambahkan code disini untuk menampilkan figure

#%%
# Membangun K-Means
algorithm = (KMeans(n_clusters =  3, init='k-means++', n_init = 10, max_iter=300, tol=0.0001, random_state= 111 , algorithm='elkan'))# isi nilai n sesuai dengan hasil elbow,init='k-means++', n_init = 10, max_iter=300, tol=0.0001, random_state= 111 , algorithm='elkan') )
algorithm.fit(X1)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

# Menyiapkan data untuk bentuk visual cluster
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_
step = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
Z1 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) # array diratakan 1D

# Melihat bentuk visual cluster

plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z1 = Z1.reshape(xx.shape)
plt.imshow(Z1 , interpolation='nearest',
extent=(xx.min(), xx.max(), yy.min(), yy.max()),
cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')
plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data= df , c = labels2 , s = 200 )
plt.scatter(x = centroids2[: , 0] , y = centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')
plt.show()
 #tambahkan code disini untuk menampilkan figure
# Melihat nilai Silhouette Score
score2 = silhouette_score(X1, labels2)
print("Silhouette Score: ", score2)


# %%
X1 = df[['Age', 'Annual Income (k$)' , 'Spending Score (1-100)']]
scaler = StandardScaler()
scaler = scaler.fit(X1)
std = scaler.transform(X1)

std = pd.DataFrame(std)

std.rename(inplace = True, columns = {0:"Std Age",
1:"Std Annual Income",
2:"Std Spending Score"})

std.tail()

kmeans = KMeans(n_clusters = 4)
kc = kmeans.fit_predict(std)

centroids = pd.DataFrame(kmeans.cluster_centers_)
centroids. rename(inplace = True, columns = {0:"Std Age",
1:"Std Annual Income",
2:"Std Spending Score"})

cluster = pd.DataFrame(kc)
df_cluster = pd.concat([X1, cluster], axis = 1).rename(columns = {0:"Cluster"})

df_cluster.tail() # database preview
df_cluster.groupby("Cluster").count() # distribution by clusters
centroids # centroids preview

fig = plt.figure(figsize = (15,14))
ax = fig.add_subplot(111, projection='3d')

ax.set(xlabel = "Std Age",
ylabel = "Std Annual Income",
zlabel = "Std Spending Score")

x = std["Std Age"]
y = std["Std Annual Income"]
z = std["Std Spending Score"]

ax.scatter(x,y,z, marker="o", c = kc, s=150, cmap="brg", edgecolors= "black")

ax.scatter(centroids["Std Age"],
centroids["Std Annual Income"],
centroids["Std Spending Score"],
marker = "p", s = 300, c = "black")

ax.view_init(30,40)
# %%
