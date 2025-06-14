# preparation
## Importeren van libraries
```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
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

## Split data in TRAIN and TEST sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, stratify=y) # Stratify is used so the classes for the tree are evenly spread as well
```

# Model Selection
## Creating Tree Classifier
```python
# Random state gebruikt omdat we allemaal dan dezelfde values krijgen
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

model = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=5, random_state=42) # Deze opties zijn gewoon een voorbeeld, mogen weg BEHALVE gini

# Training van het model
model.fit(X_train, y_train)

# Tonen van het model
print(f"Model classes: {model.classes_}")
plot_tree(model)
```

## Validate Model 
```python
# importeren libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# predictions op basis van het model zijn predictors
y_test_pred = model.predict(X_test)  

# VALIDATIONS
# Acc geeft aan hoe vaak hij juist is
acc = accuracy_score(y_true=y_test, y_pred=y_test_pred)  
prec = precision_score(y_true=y_test, y_pred=y_test_pred, average='weighted')  
rec = recall_score(y_true=y_test, y_pred=y_test_pred, average='weighted')  
f1 = f1_score(y_true=y_test, y_pred=y_test_pred, average='weighted')  

# Geeft de validatie voor het hele model
print(f'ACC : {acc:.3f} - PREC : {prec:.3f} - REC : {rec:.3f} - F1 : {f1:.3f}')  

# De validatie voor de confusion matrix
print(classification_report(y_true=y_test, y_pred=y_test_pred))
```
**Accuracy** score geeft aan hoe vaak hij het juist heeft – Stel model met klassen hond, kat en muis hoe vaak hij deze juist clasifiseert is de accuracy, de precision is hoe vaak de hond juist is gegokt gedeeld door hoe vaak de hond gegokt is

Stel 8 keer juist hond gegokt, 2 keer gedacht dat het een hond was maar was kat of muis 8/10 = 0.8 = 80%

**Recal** is hoevaak hond juist gegokt en hoevaak hond fout gegokt (dan muis of kat voorspeld) stel 8 keer juist en 12 keer gedacht dat het een kat of muis was dan 8/20 = 0.4 = 40%

**F1 score** neemt een gemiddelde van de percision en de recall, dit kunnen we doen per klasse en van het hele model. Voor heel het model doen we van iedere klasse en dan het gemiddelde

We kijken naar de **accuracy** om te zien hoe goed een model is, hoe hoger deze hoe beter, ook kijken we naar de **macro** en **weighted averages**, hoe hoger de **f1** hoe beter

### Evaluate function
Deze functie kan makkelijk gebruikt worden om alle voorgande tests uit te voeren, verwacht enkel de y_test en de y_test_pred sets
```python
def evaluate(y_test, y_test_pred):  
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report  
    acc = accuracy_score(y_true=y_test, y_pred=y_test_pred)  
    prec = precision_score(y_true=y_test, y_pred=y_test_pred, average='weighted')  
    rec = recall_score(y_true=y_test, y_pred=y_test_pred, average='weighted')  
    f1 = f1_score(y_true=y_test, y_pred=y_test_pred, average='weighted')  
  
    print(f'ACC : {acc:.3f} - PREC : {prec:.3f} - REC : {rec:.3f} - F1 : {f1:.3f}')  
  
    print(classification_report(y_true=y_test, y_pred=y_test_pred))
```

## Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Gewone simpele array weergave
print(confusion_matrix(y_test, model.predict(X_test)))

# Plot confusion matrix (nicer output in Jupyter) 
ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_test_pred, display_labels=model.classes_)
```

## Toepassen op nieuwe data
```python
X_pred = # nieuwe feature die we als predictor gebruiken
y_pred = model.predict(X_pred) # voorspelde data
```

