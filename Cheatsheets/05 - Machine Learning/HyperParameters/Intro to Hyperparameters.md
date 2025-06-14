There are **2 Types** of machine learning 
- Supervised
	- Geven we de oplossing mee 
	- Kan getallen/continue variabelen bevatten of categorical zijn
- Unsupervised
	- Gaat zelf patronen zoeken

Volgen altijd 6 stappen
1. Data inladen
2. Model selecteren
3. Model trainen
4. Model tonen
5. Model valideren
6. Model toepassen op nieuwe data

## Cross validation
Hier gaan we de data die we hebben opsplitsen in verschillende blokken en iedere blok 1x als test-data gebruiken.  Deze blokken noemen we ook vaults
Kan in 2 blokken
![[Pasted image 20250516214955.png]]

Maar ook in 5
![[Pasted image 20250516215016.png]]
we kunnen ze in n-aantal stukken verdelen en zullen dan n-aantal keer valideren

### Code
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
```

We geven deze welk model we maar hebben, de inputs, verwachte outputs en de cross-validation vault (in hoeveel stukken hij deze moet kappen)
-> Geeft een score terug (is model dependent kan accuracy zijn voor decision trees en voor linear regression)

## underfitting en overfitting
Goeie visuele weergaven hiervan is
![[Pasted image 20250516215322.png]]
Links zien we underfitting en rechts overfitting
Elke top en dal wijst op een extra graad in de formule - hoe complexer ons model hoe meer mogelijkheden hij krijgt

Links is gewoon een lineare regressie, heeft 2 parameters - hier is sprake van underfitting - te weinig parameters dus model is te **simpel**

Rechts is polynomiale regressie, enorm veel bochten dus veel kwadraten - sprake van overfitting - model is te **complex**

In deze figuur zijn de blauwe punten de test data en de rode de validatie 

### Oplossing
![[Pasted image 20250516215632.png]]
Kan op 2 manieren *Minder parameters*/*Meer data*

## Model Selection and Hyperparameter Tuning
Om eigenlijk het perfecte model te vinden zouden we onze data nog eens moeten opspliten in - 1 training set - 1 Validatie set - 1 test set

Deze test set zouden we dan gebruiken om te zien of het wel het optimale model is

Dit neemt veel data in beslag dus we kunnen hier een andere methode voor gebruiken
- We gaan de training en validatieset samen nemen en nemen enkel een test-set gebruik makende van cross-validatie
### Code
```python
from sklearn.model_selection import validation_curve
Train_scores, val_scores = validation_curve(model, X, y, cv=5, param_name, param_range)
```

Hier geven we dan ook eender welk model aan mee - X is onze feature, y is wat we willen voorspellen. parameter_name is de naam van de parameter IN HET MODEL en parameter range is de graad waarin hij deze parameter zal testen. 

Stel we geven een parameter range van n-.arange(1, 6) = (1, 2, 3, 4, 5) - Dan zal hij deze graden testen MET cross-validation dus hij zal graad 1 doen met een cross-val van 5, dan 2 met cross-val van 5 en zo voort dus zal uiteindelijk 25 keer uitgevoerd worden

Deze range hoeft niet een range van getallen te zijn, het kan bijvoorbeeld ook de optie voor een intercept zijn (True, False) dan zal hij een keer het beste model proberen vinden voor zowel True als False

met `model.get_params()` krijgen we alle parameters die we voor een model kunnen gebruiken

## Meerdere parameters
Hiervoor gebruiken we een grid search

```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(model, cv=5, param_grid=param)
```

*model* - geven we eender welk model
*cv* - aantal cross-validations dat we willen doen
*param_grid* - een dictionary met parameters {'criterion': \['gini', 'entropy'], 'max_depth': np.arange(1,6)} 

Bijvoorbeeld bij een decision tree kunnen we zowel een diepte als criteria meegevn
- Hier geven we gini en entropy met een diepte van 1-6 (6 niet mee) met een cv van 5 dus deze zal 10 parameters 5x cross-valideren dus er zullen 50 trainings gebeuren

grid.fit zal trainen *best_params_* toont gemiddeld de beste parameters en *best_score_* toont gemiddeld de beste score

*grid.best_estimator_* geeft het beste model welke we kunnen gebruiken om voorspellingen te doen

### Compleet script
```python
from sklearn.model_selection import GridSearchCV

grid_param = {'criterion': ['gini', 'entropy', 'log_loss'],
			   'max_depth' : list(range(2, 10)),
			   'min_samples_split' : list(range(2, 5))}

grid_search = GridSearchCV(model, grid_param, cv=5)

grid_search.fit(X_tr, y_tr)

print(f'Best parameters : {grid_search.best_params_}') 
print(f'Best score  : {grid_search.best_score_:.3f}')
```