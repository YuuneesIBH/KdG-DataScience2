# regression.py
import pandas as pd
from sklearn.linear_model import LinearRegression               # model: Ordinary Least Squares
from sklearn.model_selection import train_test_split            # splitsen in train/test
from sklearn.metrics import r2_score, mean_squared_error        # evaluatie metrics

def multiple_linear_regression(
    data: pd.DataFrame,
    features: list,
    target: str,
    test_size: float = 0.2,
    random_state: int = None
) -> dict:
    """
    Voert een lineaire regressie uit met meerdere features.
    
    Parameters:
    - data        : DataFrame met alle kolommen
    - features    : lijst van voorspellende kolomnamen (list of str)
    - target      : naam van de doelvariabele kolom (str)
    - test_size   : fractie van data in testset (0.0–1.0)
    - random_state: seed voor reproduceerbaarheid

    Returns:
    dict met:
    - model : getraind LinearRegression-object
    - X_test: DataFrame met test-kenmerken
    - y_test: Series met werkelijke target-waarden
    - y_pred: array met voorspelde waarden
    - r2    : R²-score op testset
    - rmse  : root-mean-squared error op testset
    """
    # 1. Data voorbereiden
    X = data[features]               # DataFrame met meerdere features
    y = data[target]                 # Series met target-variabele
    
    # 2. Train/test-split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 3. Model trainen
    model = LinearRegression()       # intercept en coëfficiënten bepalen
    model.fit(X_train, y_train)
    
    # 4. Voorspellen op testset
    y_pred = model.predict(X_test)
    
    # 5. Evaluatie
    r2   = r2_score(y_test, y_pred)                          # goodness of fit
    rmse = mean_squared_error(y_test, y_pred, squared=False) # standaardfout
    
    return {
        "model":   model,
        "X_test":  X_test,
        "y_test":  y_test,
        "y_pred":  y_pred,
        "r2":      r2,
        "rmse":    rmse
    }
