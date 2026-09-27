# src/model.py

import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------------------------

def build_logistic_model():

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    return model


# ---------------------------------------------------------
# RANDOM FOREST
# ---------------------------------------------------------

def build_random_forest():

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    return model


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

def train_model(model, X_train, y_train):

    model.fit(
        X_train,
        y_train
    )

    return model


# ---------------------------------------------------------
# PREDICT PROBABILITY
# ---------------------------------------------------------

def predict_probabilities(model, X):

    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------
# SAVE / LOAD
# ---------------------------------------------------------

def save_model(model, path):

    joblib.dump(
        model,
        path
    )


def load_model(path):

    return joblib.load(path)