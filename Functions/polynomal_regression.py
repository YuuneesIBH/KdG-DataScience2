# Import libraries for data handling and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import scikit-learn modules
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Task 3: Definieer de polynomial_regression functie
def polynomial_regression(degree=2, **kwargs):
    """
    Creëert een polynomial regression model met pipeline
    
    Parameters:
    - degree: graad van de polynoom (default=2)
    - **kwargs: extra parameters voor LinearRegression
    """
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(**kwargs)
    )