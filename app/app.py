import mlflow
from flask import Flask, request, jsonify

tracking_uri = "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(tracking_uri)

class Model:
    def __init__(self):
        model_name = "model-iris-classifications"
        stage = "Staging"
        model_uri = f"models:/{model_name}/{stage}"
        self.model = mlflow.sklearn.load_model(model_uri)

    def post(self):
        data = request.get_json()
        feature_names = [
            "sepal_length", 
            "sepal_width", 
            "petal_length", 
            "petal_width"
        ]
        X = [[
            data.get(feature_name, 0) for feature_name in feature_names
        ]]
        
        pred = self.predict(X)
        return_dictionary = {
            "class_prediction": pred[0]
        }
        return jsonify(return_dictionary), 201
    
    def predict(self, X):
        return self.model.predict(X)


if __name__ == '__main__':
    app = Flask(__name__)
    service = Model()
    app.add_url_rule('/posts', view_func=service.post, methods=['POST'])
    app.run(port=5001)
