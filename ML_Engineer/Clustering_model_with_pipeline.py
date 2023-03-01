import numpy as np
import pandas as pd

import sqlite3

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import pickle

# Import data from SQLite database and convert to pandas DataFrame
def get_data_from_sqlite_db():
    connexion = sqlite3.connect('churn_prediction\BankChurners.db')
    query = "SELECT * FROM client"
    df = pd.read_sql_query(query, connexion)
    connexion.close()
    return df

df = get_data_from_sqlite_db()

# Predictive features
features = ['Total_Trans_Amt','Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Total_Amt_Chng_Q4_Q1', 
            'Total_Revolving_Bal', 'Credit_Limit', 'Months_on_book', 'Total_Relationship_Count', 
            'Avg_Utilization_Ratio', 'Contacts_Count_12_mon', 'Months_Inactive_12_mon', 'Dependent_count']

# Isolate the target from the predictive features
X = pd.concat([df[features]], axis=1)
y = np.where(df["Attrition_Flag"] == "Attrited Customer", 1, 0)

# Pipeline including preprocessing of numerical and categorical features and clustering model
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
    ('model', KMeans(n_clusters=6, n_init='auto'))])

pipe.fit(X)

# Get the predicted cluster labels and add them to the data frame
labels = pipe.predict(X)
X_labeled = X.assign(Cluster=labels)

# Add the 'Attrition_Flag' colum to the labeled data frame
X_labeled = pd.concat([X_labeled, df['Attrition_Flag']], axis=1)

# Calculate the churning rate and size of each cluster
def calculate_churning_rate_cluster(X_labeled):
    churning_proportion = X_labeled.groupby('Cluster').apply(lambda x: (x['Attrition_Flag'] == 'Attrited Customer').mean()*100)
    cluster_size = X_labeled.groupby('Cluster').size()
    proportions = pd.concat([churning_proportion, cluster_size], axis=1)
    proportions.columns = ['Churning Proportion', 'Cluster Size']
    return proportions

proportions = calculate_churning_rate_cluster(X_labeled)
print(proportions)

# Evaluate the clustering model
def evaluate_model(pipe, X):
    silhouette = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    print(f'Silhouette score: {silhouette:.3f}')
    print(f'Davies-Bouldin index: {db:.3f}')
    print(f'Calinski-Harabasz index: {ch:.3f}')

evaluate_model(pipe, X)

# Save the clustering model as a pickle file & put as a comment to avoid overwriting it afterwards
# pickle.dump(pipe, open('churn_prediction\Clustering_model_pipeline.pkl','wb'))

# Save database for dashboard in Tableau & put as a comment to avoid overwriting it afterwards
# X_labeled.to_csv('churn_prediction\BankChurnerswithCluster.csv', index=False)