from flask import Flask , request , jsonify
import numpy as np
import joblib
app= Flask(__name__)

model = joblib.load("iris_model.pkl")

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route("/predict", methods = ["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        features = data["features"]
    except KeyError:
        return jsonify({"error": "Missing 'features' in request"}), 400
    features = np.array(features).reshape(1,-1)
    prediction  = model.predict(features)
    classes = ["Setosa", "Versicolor", "Verginica"]
    result = classes[prediction[0]]
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)