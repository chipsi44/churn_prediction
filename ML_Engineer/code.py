import numpy as np 
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sqlite3


########################################################################
# Check approaches for unbalanced datasets
########################################################################


# Connect to the SQLite database and extract data from it into a pandas DataFrame
connexion = sqlite3.connect('./BankChurners.db')
query = "SELECT * FROM client"
df = pd.read_sql_query(query, connexion)
connexion.close()

# Replacing 'Unknown' values by None
df.replace('Unknown', None, inplace=True)
# df.dropna() -- à voir si les colonnes concernées sont impliquées dans le modèle

""" 
df.columns = ['CLIENTNUM', 'Attrition_Flag' , 'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
       'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 
       'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 
       'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',]
"""

# Columns to remove from the predictives features -- to be completed after EDA
columns_to_drop = ['Attrition_Flag', 'CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']

# Isolate the feature to predict from the predictive features
X = df.drop(columns_to_drop, axis=1) 
y = df['Attrition_Flag']

# Label encode object columns from database
def label_encode(df):
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(X[object_cols], columns=object_cols, prefix=object_cols, drop_first=False)
    df = pd.concat([df, df_encoded], axis=1)
    df.drop(object_cols, axis=1, inplace=True)
    return df

X = label_encode(X)

# Labeling the target
y.replace({'Existing Customer': 0, 'Attrited Customer': 1}, inplace=True)

# Split the database into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale the features
def scale_features(X_train, X_test):
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# Train the model
def model(model, X_train, y_train):
    trained_model = model.fit(X_train,y_train.values)
    return trained_model

trained_model = model(LogisticRegression(solver='lbfgs'), X_train_scaled, y_train)

print(trained_model.score(X_test, y_test))