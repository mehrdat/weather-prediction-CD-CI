from flask import Flask,jsonify,request
import joblib
import pandas as pd
import numpy as np
import sys
import traceback

from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor as gbr
from sklearn.feature_selection import SelectKBest,mutual_info_classif,mutual_info_regression
from sklearn.metrics   import f1_score
from sklearn.metrics import mean_squared_error,r2_score
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor


df=pd.read_csv('./data/weth001.csv',parse_dates=['date'])

df['year']=df.date.dt.year
df['month']=df.date.dt.month


df=df.apply(lambda x: x.replace(' ',np.nan))

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer,MissingIndicator,MissingIndicator,IterativeImputer,MissingIndicator


dtypes={'rain':float,'temp':float,'wetb':float,'dewpt':float,

        'vappr':float,'rhum':float,'msl':float,
        'wdsp':float,'wddir':float}#,'sun':float,'vis':float,'clht':float,'clamt':float}

df=df.astype(dtypes)

df=df.select_dtypes(include=['number'])

df=df.drop(columns=['Unnamed: 0','latitude','longitude'])
it=IterativeImputer()
df=pd.DataFrame(it.fit_transform(df),columns=df.columns)


df=df.drop(['rain','msl','wdsp','wddir','vappr'],axis=1)


from sklearn.model_selection import train_test_split
y=df['temp']
X=df.drop('temp',axis=1)



#print(X[-4:].to_dict(orient='records'))

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.4, random_state=42)


#df=df.reset_index()

#df.drop([''])
# data_train_proph=data_train.reset_index().rename(columns={'date':'ds','temp':'y'})
# data_test_proph=data_test.reset_index().rename(columns={'date':'ds','temp':'y'})
# data_test_predicted=model.predict(data_test_proph)



print("so far so good")

print(df.head())

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error


# ExtraTreesRegressor

# model = ExtraTreesRegressor(random_state=42, n_estimators=200,max_depth=20)
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)


# XGBRegressor
# model = XGBRegressor(random_state=42, n_estimators=200,max_depth=20)
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)


train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data])
y_pred = model.predict(X_test, num_iteration=model.best_iteration)


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#print("R2: ",r2_score(y_test, y_pred))


joblib.dump(model, 'model.pkl')

print('Model saved')

lr = joblib.load('model.pkl')

model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')


