import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# LOAD DATA

data = pd.read_csv('data/Placement_Data.csv')

# Drop unnecessary columns
data.drop(['sl_no', 'salary'], axis=1, inplace=True)

# Features & target
X = data.drop('status', axis=1)
y = data['status']


# COLUMN TYPES

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns


# PREPROCESSING PIPELINE

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# MODELS

models = {
    "logistic": LogisticRegression(max_iter=500),
    "rf": RandomForestClassifier()
}

# Hyperparameter tuning for Random Forest
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20]
}

best_model = None
best_score = 0


# TRAINING LOOP

for name, model in models.items():
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    if name == "rf":
        grid = GridSearchCV(pipeline, param_grid, cv=3)
        grid.fit(X, y)
        model_final = grid.best_estimator_
    else:
        pipeline.fit(X, y)
        model_final = pipeline

    score = model_final.score(X, y)
    print(f"{name} accuracy: {score}")

    if score > best_score:
        best_score = score
        best_model = model_final


# SAVE MODEL

joblib.dump(best_model, 'model/model.pkl')

print("Model trained and saved successfully!")