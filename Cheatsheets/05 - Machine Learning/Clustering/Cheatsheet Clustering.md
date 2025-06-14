# preparation
## Importeren van libraries
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

## Inladen en tonen data
```python
data = pd.read_csv('../../datasets/file.csv', sep=",", decimal=".", header=None)

data.dropna() # als we unknown data willen verwijderen
data.drop(columns=['column_name']) # indien we een kolom willen verwijderen
data.head()
data.describe()
type(data)
```

## Data preparation
```python
# Target - Wat we willen voorspellen
y = data['feature'] # gewoon een kolom die we willen voorspellen
# Predictors - waarmee we dat gaan voorspellen
X = data[['predict_feature1', 'predict_feature2']]
```

## Exploring the data
```python
print(y.info())
print(X.info())

print(y.head())
print(X.head())

print(y.describe())
print(X.describe())
```

## (OPTIONEEL) normaliseren dataset
Hier gaan we met Z-scores werken om de uitschieters weg te werken
- Z-score is hoeveel standaardafwijkingen ze van het gemiddelde afzitten Z-score van 1 is 1 standaardafwijking van het gemiddelde
```python
scaler = StandardScaler() / MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns) # Zal van alle features de Z-score nemen
```

# Model Selection
## Initialise the model
```python
# Kunnen deze gewoon zo initialiseren
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, random_state=42)

# Of als we met genormaliseerde data werken kunnen we een pipeline gebruiken
kmeans_pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=42))
```

## model trainen
```python
# We zullen een model trainen en meteen de clusters hieruit halen
clustes = kmean.fit_predict(X)

# Of we kunnen dit van de pipeline
clusters = kmeans_pipelin.fit_predict(X)
```
## Get cluster centers and labels
```python
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
```

## Compare clusters to original data
```python
compare = pd.DataFrame({'Original': y, 'Clusters': kmeans})
compare
```
Hier zien we welke waarde van de originele data in welk cluster wordt voorspeld door het model

## Visualise the clusters
```python
# als we enkel de punten van de clusters willen plotten
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)

# Als we een dataframe hebben en deze volledig willen plotten
plt.figure(figsize=(8,6))

# Plot the points and color the cluster
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="viridis", s=100, alpha=0.6, edgecolor="k")

# plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids') 
plt.title('KMeans Clustering')

plt.legend() 
plt.show()
```

## ZIE OEFENINGEN 05.11 VOOR MEER INFO EN OEFENINGEN