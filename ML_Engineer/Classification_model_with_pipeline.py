import numpy as np
import pandas as pd
import pickle

from imblearn.over_sampling import RandomOverSampler

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

from lightgbm import LGBMClassifier

import sqlite3

# Import data from SQLite database and convert to pandas DataFrame
def get_data_from_sqlite_db():
    connexion = sqlite3.connect('churn_prediction\BankChurners.db')
    query = "SELECT * FROM client"
    df = pd.read_sql_query(query, connexion)
    connexion.close()
    return df

df = get_data_from_sqlite_db()

# Predictives features
features = ['Total_Trans_Amt','Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Total_Amt_Chng_Q4_Q1', 'Total_Revolving_Bal', 
            'Credit_Limit', 'Months_on_book', 'Total_Relationship_Count', 'Avg_Utilization_Ratio', 
            'Contacts_Count_12_mon', 'Months_Inactive_12_mon', 'Dependent_count']

# Isolate the feature to predict from the predictive features
X = pd.concat([df[features]], axis=1)
y = np.where(df["Attrition_Flag"] == "Attrited Customer", 1, 0)

# Split  database into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Oversample minority category
def dataset_balancing(X, y):
    oversample = RandomOverSampler(sampling_strategy="not majority")
    X_balanced, y_balanced = oversample.fit_resample(X, y)
    return X_balanced, y_balanced

X_train, y_train = dataset_balancing(X_train, y_train)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy='mean')), 
           ("scaler", StandardScaler())]
)

categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy='most_frequent')),
           ("encoder", OneHotEncoder(handle_unknown="ignore"))]
)
preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features),
                  ("cat", categorical_transformer, categorical_features)]
)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier())])

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

# Saving model as a pickle file
pickle.dump(pipe, open('churn_prediction\Classification_model_pipeline.pkl','wb'))

def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Not churning", "Churning"]))

evaluate_model(pipe, X_test, y_test)

# Prediction - input is pandas.DataFrame of shape (1,12)
# y_df = pd.DataFrame(y)
# predict123 = pipe.predict(X.iloc[[6292], :])
# result123 = y_df.iloc[6292,0]
# print(predict123, result123)