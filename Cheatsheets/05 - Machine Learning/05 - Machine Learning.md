## Supervised learning
Hier geven we de juiste antwoorden zodat de machine kan kijken hoe vaak hij het juist heeft

**Regressie**: Hier gaan we getallen voorspellen - bijvoorbeeld hoeveel ijs wordt er verkocht in de zomer
- *Lineair Regression*: Voorspelt een numerische uitkomst
- *Polynomial Regression*: Voorspelt een polyonomiale functie met verschillende graden - niet lineair

### Lineaire regressie
Hier hebben we 1 **Predictor** dit is wat we willen voorspellen 
- Functie: `y = ax + b`
	Hier is a= de slope/helling
	en b= intercept - waar hij begint op y waar x = 0 -> waar hij begint
	en x is hier dan de predictor of wat we gaan voorspellen

We kunnen ook meerdere predictors nemen, dan krijgen we volgende functie
- functie: `y = b + a1x1 + a2x2 + anxn` 
Als we meerdere predictoren hebben is het geen rechte meer maar een vlak in een 3-dimensionale ruimte, meer dan 2 kunnen wij ons niet meer visueel voorstellen
![[Pasted image 20250426163315.png]]

#### In Code
In code is het redelijk eenvoudig, wij gaan een model nemen *LinearRegression* en gewoon parameters meegeven. wat we meegeven is 
- De intercept (als we willen, moet niet)
- De Slope/Helling
- De jobs - dit is hoeveel cores hij gebruikt, voor ons is 1 altijd genoeg dus laten we zo

Meestal laten we alles default behalve de intercept

### Data Voorbereiden
```python
# Data PREPARATION
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Gewoon een voorbeeld van een dataset
iris = sns.load_dataset('iris')

# Predictor - Pandas DataFrame, GEEN SERIES
X = iris[['petal_width']]
# Target feature to predict: Pandas Series
y = iris['petal_length']

# SPLIT in TRAIN en TEST dataset
# Hier nemen we voor zowel de target als predictors test en train data - nemen 80% training /20% testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
```

1. Hier laden we dus eerst een dataset in, dit kunnen we ook op andere manieren doen en het is belangrijk te zien dat als we van een csv inladen we het ook juist doen. 
2. Daarna gaan we hier de predictoren (aan de hand van wat we willen voorspellen (in dit geval de breedte van de blaadjes)) onderscheiden van de Target (wat we willen voorspellen (in dit geval de lengte van de blaadjes))
3. Nadien gaan we dit dan verdelen in test en training data en in dit geval nemen we 80% test en 20% training

### Model maken met gebruik van training data
```python
# MODEL SELECTION AND HYPERPARAMETER SELECTION (MODEL SPECIFIC)
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# List all selecter hyperparameters -> toont een lijst van mogelijke parameters die op ons model staan
print(model.get_params(deep=True)) # geeft ('copy_x'=true, 'fit_intercept'=true,'n_jobs'=none, 'positive'=false)

# DERIVE MODEL FROM LABELED DATA - Dit zal ons model trainen
model.fit(X_train, y_train)

# DISPLAY COEFFICIENTS (intercept and slope) 
print(f'Intercept: {model.intercept_}') # Waar de lijn de Y-as snijdt en X = 0
print(f'Coefficient: {model.coef_[0]}') # De slope oftewel de helling, 1 predictor dus 1 value in de lijst 
```
Hier gaan we dus eigenlijk ons model trainen - gebeurt in de model.fit methode. Wanneer dit gebeurd is is ons model klaar om getest te worden

### Model Valideren, kijken hoe goed hij is
```python
# VALIDATE MODEL USING LABELED TEST DATA
# Score model, used metric is model dependent, for regression: R^2
r2 = model.score(X_test, y_test)
print(f'R^2: {r2:.3f}')

# Other metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score

# Predict target feature for the labeled test data 
# Met ons getrained model gaan we y-waarden voorspellen a.d.h.v. de X_test waarden & deze dan door verschillende tests gooien om te zien hoe goed de voorspelling is
y_test_pred = model.predict(X_test)

# Hier zien we dan dat we telkens de werkelijke waarden gaan vergelijken met de voorspelde en dan weten we hoe goed ons model is
mae = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
mape = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred) 
print(f'MAE= {mae:.3f} - MAPE= {mape:.3f} - RMSE= = {rmse:.3f} - R^2= = {r2:.3f}')
```
Nadat ons model getrained is zullen we de resterende test-data gebruiken om deze te testen. En aan de hand van deze scores zullen we weten hoe goed of slecht ons model is.

**R2**:
- Hoe dichter bij 1 hoe dichter bij perfectie
- 0 is gewoon het gemiddelde van alle data
- negatief wilt zeggen dat hij zelfs slechter voorspelt dan dat hij gewoon het gemiddelde zou nemen = SLECHT

**MAE** (Mean Absolute Error):
- Geeft een error in getallen maar kunnen we niet zo veel mee (als we in ons model enorme waarden hebben en een error van 5 weten we niet of het een grote error is of niet) - *werkt best als we 2 modellen met elkaar gaan vergelijken, dan de kleinste error is het beste*

**MAPE** (Mean Absolute Percentage Error)
- Geeft een persentage van hoe groot de error is, - Hoe kleiner hoe beter

**RMSE** (Root Mean Squared Error)
- Geeft opnieuw een getal, ook goed om 2 modellen te vergelijken

#### stel we krijgen nog nieuwe data die we willen testen met een vorig model
```python
X_pred = nieuwe_data # Moet of numpy 2-dimensionale array zijn of DataFrame
y_pred = model.predict(X_pred) # En dan kunnen we deze gewoon weer valideren
```

### Resultaat tonen in SCATTERPLOT
```python
plt.figure(figsize=(8,6))
sns.regplot(x=X_train, y=y_train, scatter_kws={"color":"blue"}, line_kws={"color":"red"})
plt.title("Linear Regression")
plt.xlabel("X values")
plt.ylabel("y values")
plt.show
```

### Model trainen met meerdere predictoren
Verandert eigenlijk heel weinig, enkel het inladen van de X
```python
# DATA PREPARATION
X = iris[['sepal_width', 'sepal_length', 'petal_width']] # 3 predictoren - 3 coefficienten
y = iris['petal_length'] # Wat we willen voorspellen

# Model selecteren
model = LinearRegression()

# SPLIT in TRAIN and TEST dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# Model trainen
model.fit(X_train, y_train)

# Coefficienten tonen
print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_}') 
```

Zoals we zien verandert er niet veel, nu gaan we wel meerdere coefficienten terugkrijgen

Dit zouden we op exact dezelfde manier valideren als we hierboven gezien hebben

### Polynomial Regressie
Hier gaan we geen rechte lijn meer hebben maar zullen we maxima en minima krijgen

```python
# MODEL SELECTION
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False, LinearRegression())) # degree zullen we aanpassen en wat moeten zien welke het beste is

# Trainen van model
model.fit(X_train, y_train)

# Valideren
r2 = model.score(X_test, y_test)

# Apply model on new Data
X_pred = ... # new data to predict
y_pred = model.predict(X_pred) # Deze kunnen we dan gebruiken om te gaan valideren
```