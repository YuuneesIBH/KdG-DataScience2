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

## Split data in TRAIN and TEST sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, stratify=y) # Stratify is used so the classes for the tree are evenly spread as well
```

# Model Selection
## Creating and validating random forest model
```python
from sklearn.ensemble import RandomForestClassifier

# Aanmaken van model met parameters
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Trainen van model
rf.fit(X_train, y_train)
```

## Make predictions
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# predictions op basis van het model zijn predictors
y_test_pred = rf.predict(X_test)  

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
print(confusion_matrix(y_test, rf.predict(X_test)))

# Plot confusion matrix (nicer output in Jupyter) 
ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_test_pred, display_labels=rf.classes_)
```