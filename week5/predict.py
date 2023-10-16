import pickle

from flask import Flask, request, jsonify

model_file = "model_C=1.0.bin"

with open(model_file, "rb") as f_in:
    dv, model1 = pickle.load(f_in)

app = Flask("approve")

@app.route("/predict", methods=["POST"])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model1.predict_proba(X)[0, 1]
    poutcome = y_pred >= 0.5

    result = {
        "approve_probability": float(y_pred),
        "poutcome": bool(poutcome)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

