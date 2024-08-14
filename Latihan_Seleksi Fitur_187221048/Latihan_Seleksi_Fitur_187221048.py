
#%%

import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Load dataset
iris = load_iris()
x, y = iris.data, iris.target

# Apply SelectKBest class to extract top 2 best features
best_features = SelectKBest(score_func=chi2, k=2)
x_new = best_features.fit_transform(x, y)

# Show selected features
print(x_new.shape)
print(x_new)

#%%

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#Load dataset
iris = load_iris()
x = iris.data
y = iris.target

# Apply PCA
pca = PCA(n_components=2) # Menentukan jumlah komponen yang diinginkan
x_pca = pca.fit_transform(x)
print (x_pca)

#Plot hasil PCA
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of IRIS Dataset')
plt.colorbar(label='Species')
plt.show()

#%%

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

# Buat model Decision Tree
model = DecisionTreeClassifier()
# Latih model
model.fit(x, y)
# Hitung Feature Importance
importances = model. feature_importances_
#Plot Feature Importance
plt.bar(range(x.shape[1]), importances)
plt.xticks(range(x.shape[1]), iris. feature_names, rotation=90)
plt.xlabel('Fitur')
plt.ylabel('Importance')
plt. show()


#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
x = iris.data
y = iris.target

# Buat model Random Forest
model = RandomForestClassifier()

# Latih model
model.fit(x, y)

# Hitung Feature Importance
importances = model.feature_importances_

#Plot Feature Importance
import matplotlib.pyplot as plt
plt.bar(range(x.shape[1]), importances)
plt.xticks(range(x.shape[1]), iris. feature_names, rotation=90)
plt.xlabel('Fitur')
plt.ylabel('Importance')
plt. show()

#%%

from sklearn.ensemble import GradientBoostingClassifier

iris = load_iris()
x = iris.data
y = iris.target

# Buat model Gradient Boosting
model = GradientBoostingClassifier()

# Latih model
model.fit(x, y)

# Hitung Feature Importance
importances = model.feature_importances_

# Plot Feature Importance
plt.bar(range(x.shape[1]), importances)
plt.xticks(range(x.shape[1]), iris. feature_names, rotation=90)
plt.xlabel('Fitur')
plt.ylabel('Importance')
plt. show()
# %%
