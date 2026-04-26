"""
Launcher for advanced model training.
Run this instead of `python -m src.advanced_models` so that StackedRegressor /
StackedClassifier are pickled under the importable module path src.advanced_models
(required for joblib.load in app.py to work).
"""
import sys
from src.advanced_models import run

tune = "--no-tune" not in sys.argv
if not tune:
    print("Running without hyperparameter tuning (fast mode)")
run(tune=tune)
