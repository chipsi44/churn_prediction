import csv
import sqlite3

conn = sqlite3.connect('BankChurners.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE client 
(
    CLIENTNUM INTEGER PRIMARY KEY,
    Attrition_Flag TEXT,
    Customer_Age INTEGER,
    Gender TEXT,
    Dependent_count INTEGER,
    Education_Level TEXT,
    Marital_Status TEXT,
    Income_Category TEXT,
    Card_Category TEXT,
    Months_on_book INTEGER,
    Total_Relationship_Count INTEGER,
    Months_Inactive_12_mon INTEGER,
    Contacts_Count_12_mon INTEGER,
    Credit_Limit INTEGER,
    Total_Revolving_Bal INTEGER,
    Avg_Open_To_Buy INTEGER,
    Total_Amt_Chng_Q4_Q1 FLOAT,
    Total_Trans_Amt INTEGER,
    Total_Trans_Ct INTEGER,
    Total_Ct_Chng_Q4_Q1 FLOAT,
    Avg_Utilization_Ratio FLOAT ,
    Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1 FLOAT,
    Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2 FLOAT
)
''')



conn.commit()
conn.close()