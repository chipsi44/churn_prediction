import pandas as pd
import sqlite3
def csv_to_db(csv_file,db_name) :
    df = pd.read_csv(csv_file)
    conn = sqlite3.connect(db_name)
    df.to_sql('client', conn, if_exists='replace', index=False)
    conn.close()