import pyodbc as dbc
import os
from dotenv import load_dotenv

load_dotenv()

connection_params = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv("DB_SERVER")};"
    f"DATABASE={os.getenv("DB_NAME")};"
    f"UID={os.getenv("DB_USER")};"
    f"PWD={os.getenv("DB_PASSWORD")};"
)

def getConnection():
    return dbc.connect(connection_params)