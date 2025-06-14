## Importeren van librarys
```python
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
from matplotlib.pyplot import title  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.pipeline import make_pipeline
```

## Inladen en tonen Data
```python
data = pd.read_csv('../../datasets/file.csv', sep=',', decimal='.', header=None) # HAAL header=none weg indien er wel een is
data.dropna() # Zal unknown values droppen
data.head()
data.describe()
```
>[!hint]
>Open het databestand en bekijk het eerst als text foor de seperator, daarna als csv en let op of er een header aanwezig is en wat de decimal point is.

## Correlatie weergeven tussen data - toon in heatmap
```python
data.corr() # Toont welke velden het hardste afhangt van een ander veld
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
```
>[!note]
>Hoe 'warmer' de kleur hoe hoger de correlatie hoe harder deze afhankelijk zijn van elkaar. Dat wilt zeggen dat als we een voorspelling willen maken we best als predictor diegene nemen met de hoogste correlatie omdat die de andere het beste aanvoelt.

## Model trainen voor lineaire regressie 
```python
data = pd.read_csv('../../datasets/file.csv', sep=',', decimal='.', header=None)

X = data[['predictor']] # What we want to use to predict
y = data['target'] # What we want to predict

# Split all the data into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)

# Model kiezen
model = LinearRegression()

# Model trainen
model.fit(X_train, y_train)

# Show intercept and coefficient
print(f'Intercept: {model.intercept_}') 
print(f'Coefficient: {model.coef_}')
```

## Valideren
```python
# Use the model to create a prediction using the Test predictors
y_test_pred = model.predict(X_test)

# Validate everything
mae = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
mse = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
mape = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred) 
print(f'MAE= {mae:.3f} - MSE = {mse:.3f} - MAPE= {mape:.3f} - RMSE= = {rmse:.3f} - R^2= = {r2:.3f}')
```

## Tonen in scatterplot
```python
plt.figure(figsize=(8,6))
sns.regplot(x=X_train, y=y_train, scatter_kws={"color":"blue"}, line_kws={"color":"red"})
plt.title("Linear Regression")
plt.xlabel("Predictors")
plt.ylabel("Predicted Target")
plt.show
```

## Meerdere predictors
```python
# DATA PREPARATION
X = data[['predictor1', 'predictor2', 'predictor3']] # 3 predictoren - 3 coefficienten
y = data['target'] # Wat we willen voorspellen

# Model selecteren
model = LinearRegression()

# SPLIT in TRAIN and TEST dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)

# Model trainen
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

mse = mean_squared_error(y_true=y_test, y_pred=y_test_pred)  
r2 = r2_score(y_test, y_test_pred)  
print(f'MSE = {mse:.3f} - R^2= = {r2:.3f}')
```
Kijk terug naar de heat-map en zoek diegene met ook nog hoge waarden en voeg deze toe op de plaats van predictor2 en 3

## Polynomiale regressie
Eigenlijk EXACT hetzelfde enkel kiezen we in het begin niet voor Linear maar voor Polynomial
```python
# Choose which model  
model = make_pipeline(PolynomialFeatures(degree=4, include_bias=False), LinearRegression())  
  
# Train Model  
model.fit(X_train, y_train)  

# Get Predictions
y_test_pred = model.predict(X_test)  
  
# Validate predictions  
mse = mean_squared_error(y_true=y_test, y_pred=y_test_pred)  
r2 = r2_score(y_test, y_test_pred)  
print(f'MSE = {mse:.3f} - R^2= = {r2:.3f}')
```

### Extra
Als we deze willen plotten zullen we een kleine aanpassing moeten maken
Hier kiezen we één van de predictors welke we willen plotten en geven we ook aan hoeveel degrees we hebben gekozen
```python
plt.figure(figsize=(8,6))  
sns.regplot(x=X_train['Predictor'], y=y_train,  
            scatter_kws={"color":"blue"},  
            line_kws={"color":"red"},  
            order=4)  
plt.title("Polynomial Regression (Degree 4)")  
plt.xlabel("Predictors")  
plt.ylabel("Target")  
plt.show()
```