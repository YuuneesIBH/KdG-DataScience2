## Import libraries
```python
from statistics import LinearRegression
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from jupyter_server.services.events.handlers import validate_model  
from sklearn.model_selection import validation_curve  
from sklearn.linear_model import LinearRegression  
from sklearn.pipeline import make_pipeline  
from sklearn.preprocessing import PolynomialFeatures
```

## Load Data
```python
data = pd.read_csv('../../datasets/file.csv', sep=",", decimal=".", header=None)

data.dropna() # als we unknown data willen verwijderen
data.drop(columns=['column_name']) # indien we een kolom willen verwijderen
data.head()
data.describe()
type(data)
```

## Polynomial Regression Function
Polynomiale Regressie kunnen we niet zomaar importeren dus hier moeten we een functie voor gebruiken (Deze is gewoon te copy pasten in de toekomst)

```python
def polynomial_regression(degree=2, **kwargs):  
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression(**kwargs))  
    return model
```

## Target and feature
```python
X = data[['cement', 'water', 'age']] # De features die we willen gebruiken om te voorspellen
y = data['csMPa'] # Wat we willen voorspellen
```

## training en validation scores berekenen
```python
degree = np.arange(1, 6) # Aantal features/graden die we willen testen 
model = polynomial_regression() # Kan ook een ander model zijn
train_score, val_score = validation_curve(model, X, y, param_name='polynomialfeatures__degree', param_range=degree, cv=5)
```

## Scores tonen 
```python
print(f'Train mean R2 scores      : {train_score.mean(axis=1)}')  
print(f'Validation mean R2 scores : {val_score.mean(axis=1)}')
```

## Validation curve tonen
```python
plt.plot(degree, np.mean(train_score, axis=1), color='blue', label='Training score')
plt.plot(degree, np.mean(val_score, axis=1), color='red', label='Validation score')
plt.legend()
plt.xlabel('Degree')
plt.ylabel('Score')
plt.title('Validation curve for polynomial Regression')
plt.show()
```

## Grid Search
```python
from sklearn.model_selection import GridSearchCV

grid_param = {'polynomialfeatures__degree': np.arange(1, 10), 'linearregression__fit_intercept': [True, False]}
grid = GridSearchCV(model, param_grid=grid_param, cv=7)

grid.fit(X, y)

print(grid.best_params_)
print(grid.best_score_)
```