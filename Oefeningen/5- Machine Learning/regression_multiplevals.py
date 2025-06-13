# regression_multiplevals.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def multiple_linear_regression(
    data: pd.DataFrame,
    features: list,
    target: str,
    test_size: float = 0.2,
    random_state: int = None
) -> dict:
    """
    Voert een lineaire regressie uit met meerdere features.
    """
    # 1. Data voorbereiden
    X = data[features]
    y = data[target]

    # 2. Train/test-split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # 3. Model trainen
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Voorspellen op testset
    y_pred = model.predict(X_test)

    # 5. Evaluatie
    r2   = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return {
        "model":  model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "r2":     r2,
        "rmse":   rmse
    }
