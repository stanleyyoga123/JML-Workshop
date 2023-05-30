import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



tracking_uri = "http://127.0.0.1:5000/"
experiment = "iris-classifications"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment)

filename = "iris.csv"
y_label = "class"

df = pd.read_csv(filename)
feature_names = [
    "sepal_length", 
    "sepal_width", 
    "petal_length", 
    "petal_width"
]


X = df[feature_names]
y = df[[y_label]]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

params = {
    "n_estimators": 10
}

model = RandomForestClassifier(**params)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(score)

with mlflow.start_run(run_name="run1"):
    mlflow.log_param("n_estimators", 10)
    mlflow.log_metric("accuracy", score)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model-random-forest"
    )