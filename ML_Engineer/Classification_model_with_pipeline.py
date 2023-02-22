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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

# Columns to remove from the predictives features -- to be completed after full EDA
columns_to_drop = ['Attrition_Flag', 'CLIENTNUM', 'Customer_Age', 'Dependent_count', 'Marital_Status', 'Income_Category', 
        'Months_on_book', 'Avg_Open_To_Buy',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']

# Isolate the feature to predict from the predictive features
X = df.drop(columns_to_drop, axis=1) 
y = np.where(df["Attrition_Flag"] == "Attrited Customer", 1, 0)

# Oversample minority category
def dataset_balancing(X, y):
    oversample = RandomOverSampler(sampling_strategy="not majority")
    X_balanced, y_balanced = oversample.fit_resample(X, y)
    return X_balanced, y_balanced

X, y = dataset_balancing(X, y)

# Split  database into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

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
    ('model', LGBMClassifier(min_data_in_leaf=500))])

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

# Saving model as a pickle file
pickle.dump(pipe, open('churn_prediction\Classification_model_pipeline.pkl','wb'))

def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1 score: {:.4f}".format(f1))
    print("ROC AUC: {:.4f}".format(roc_auc))

evaluate_model(pipe, X_test, y_test)