from flask import Flask, jsonify, request
import joblib
import pandas as pd
import traceback
import json
# Load model and columns globally
import sys
try:
    lr = joblib.load('model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    print('Model and columns loaded successfully.')
except Exception as e:
    print(f"Error loading model or columns: {e}")
    lr = None
    model_columns = None

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Weather Prediction API. Use the /predict endpoint to get predictions."


@app.route('/predict', methods=['POST','GET'])
def predict():
    print("welcome to the predict")
    if lr is not None:
        print("we are in the if section: if lr is not None: ")
        try:
            print("we are in the try section ")

            json_ = request.json
            
            print( "let's check the json file")
            if isinstance(json_, dict):
                print('that was right!')
                json_ = [json_]
            
            query_df = pd.DataFrame(json_)
            query = pd.get_dummies(query_df)
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = lr.predict(query)
            prediction = prediction.tolist()
            return jsonify({'prediction': prediction})

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        return jsonify({'error': 'Model not loaded. Please check the server logs for details.'})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345

    app.run(debug=True, port=port)
