# implementing a web service that will use our model to return predictions
import pickle
from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb

model_file = 'xgboost_model.bin'

with open(model_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

app = Flask('dropouts')

@app.route('/predict', methods=['POST'])
def predict():
    student = request.get_json()
    X = dv.transform([student])

    X_xgb = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    y_pred = float(model.predict(X_xgb)[0])
    dropout_decision = y_pred >= 0.5

    result = {
        'dropout_probability': y_pred,
        'dropout_decision' : bool(dropout_decision)
    }
    return jsonify(result)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696



