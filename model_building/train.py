import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

numeric_features =  "CustomerID", "ProdTaken", "Age", "TypeofContact",
    "CityTier", "DurationOfPitch", "Occupation", "Gender","NumberOfPersonVisiting", 
    "NumberOfFollowups","ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", 
    "Passport", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome", "ProdTaken"
categorical_features = ["Type"]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Small grid so the pipeline runs fast on GitHub Actions.
# Widen this if you want a more thorough search.
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

pipeline = make_pipeline(preprocessor, model)
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1)
grid.fit(Xtrain, ytrain)

best_model = grid.best_estimator_
print("Best params:", grid.best_params_)
print(classification_report(ytest, best_model.predict(Xtest)))

# Save next to app.py so the Streamlit app can load it directly
joblib.dump(best_model, "deployment/best_tourism.joblib")
print("Model saved to deployment/best_tourism.joblib")
Writing project/model_building/train.py
