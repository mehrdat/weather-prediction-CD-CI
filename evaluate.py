import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer,MissingIndicator,MissingIndicator,IterativeImputer,MissingIndicator
import numpy as np


def load_data(file_path, model_features):
    #data = pd.read_csv(file_path)
    
    # Convert non-numeric columns to numeric
    #for column in data.columns:
    #    if data[column].dtype == 'object':
    #        data[column] = LabelEncoder().fit_transform(data[column])
    
    
    df=pd.read_csv(file_path,parse_dates=['date'])

    df['year']=df.date.dt.year
    df['month']=df.date.dt.month


    df=df.apply(lambda x: x.replace(' ',np.nan))


    dtypes={'rain':float,'temp':float,'wetb':float,'dewpt':float,

            'vappr':float,'rhum':float,'msl':float,
            'wdsp':float,'wddir':float}#,'sun':float,'vis':float,'clht':float,'clamt':float}
    df=df.astype(dtypes)
    df=df.select_dtypes(include=['number'])
    df=df.drop(columns=['Unnamed: 0','latitude','longitude'])
    it=IterativeImputer()
    df=pd.DataFrame(it.fit_transform(df),columns=df.columns)
    df=df.drop(['rain','msl','wdsp','wddir','vappr'],axis=1)


    
    # Ensure the data has the same features as the model was trained on
    #data = data[model_features]
    
    return df

def evaluate_model(model_path, data_path):
    model = joblib.load(model_path)
    
    # Get the feature names from the LightGBM model
    if isinstance(model, lgb.Booster):
        model_features = model.feature_name()
    else:
        raise ValueError("The loaded model is not a LightGBM model.")
    
    data = load_data(data_path, model_features)
    
    X = data.drop('temp', axis=1)
    y = data['temp']
    
    predictions = model.predict(X)
    
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"Model Evaluation:\nMSE: {mse}\nR^2: {r2}")
    return mse, r2

if __name__ == "__main__":
    model_path = 'model.pkl'
    data_path = './data/weth001.csv'
    evaluate_model(model_path, data_path)
