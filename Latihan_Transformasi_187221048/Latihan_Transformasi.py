import pandas as pd
from sklearn import preprocessing
import numpy as np

data = pd.read_csv("shopping_data.csv")
data_angka = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].select_dtypes(include=[np.number])
df = pd.DataFrame(data_angka)
df.head()

#normalisasi dengan rumus manual
print("CARA MANUAL")
df["Age"] = df["Age"] / df["Age"].max ()
df["Annual Income (k$)"] = df["Annual Income (k$)"] / df["Annual Income (k$)"].max ()

df ["Age"] = (df["Age"] - df["Age"].min ()) / (df["Age"] .max () - df ["Age"].min ())
df["Annual Income (k$)"] = (df["Annual Income (k$)"] - df["Annual Income (k$)"].min()) / (df["Annual Income (k$)"].max () - df["Annual Income (k$)"].min())

df["Age"] = (df ["Age"] - df["Age"].mean ()) / df ["Age"].std ()
df["Annual Income (k$)"] = (df["Annual Income (k$)"] - df["Annual Income (k$)"] .mean ()) / df ["Annual Income (k$)"].std()
print(df["Age"])
print(df["Annual Income (k$)"])

#normalisasi dengan package minmaxscale
print("CARA DENGAN PACKAGE")
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
normalisasi_df = pd.DataFrame(np_scaled)
print(normalisasi_df)

#%%
import pandas as pd
from scipy.stats import zscore

# Load the Excel data into a pandas DataFrame
df = pd.read_csv('diabetes.csv')

# Calculate the z-score normalization of the data
df_zscore = df.apply(zscore)

# Print the normalized DataFrame
print(df_zscore)

# %%
