import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # Convert non-numeric columns to numeric
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = LabelEncoder().fit_transform(data[column])
    
    return data

def evaluate_model(model_path, data_path):
    model = joblib.load(model_path)
    data = load_data(data_path)
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    predictions = model.predict(X)
    
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"Model Evaluation:\nMSE: {mse}\nR^2: {r2}")
    return mse, r2

if __name__ == "__main__":
    model_path = 'model.pkl'
    data_path = './data/weth001.csv'
    evaluate_model(model_path, data_path)
