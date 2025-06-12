# validation.py
from sklearn.metrics import r2_score, mean_squared_error  # evaluatie‐metrics

def evaluate_regression(y_true, y_pred):
    """
    Bereken en toon de R²-score en Mean Squared Error voor een regressie.
    
    Parameters:
    - y_true: werkelijke waarden (array‐achtig)
    - y_pred: voorspelde waarden (array‐achtig)
    
    Returns:
    dict met:
    - r2  : R²-score
    - mse : Mean Squared Error
    """
    # R² berekenen
    r2  = r2_score(y_true, y_pred)  
    # MSE berekenen
    mse = mean_squared_error(y_true, y_pred)  

    # Resultaten afdrukken
    print(f"R²-score: {r2:.3f}")  
    print(f"MSE:      {mse:.3f}")  

    return {"r2": r2, "mse": mse}