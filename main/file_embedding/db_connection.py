from dotenv import load_dotenv, find_dotenv
import pyodbc
import os

# Resolve .env by walking up from this file — works regardless of which
# directory is used as working directory when launching a script.
load_dotenv(find_dotenv())


def get_connection():
    """Return an open pyodbc connection to the configured SQL Server database.

    Required environment variables (set in .env):
        DB_SERVER   — SQL Server hostname or IP (e.g. "localhost" or "MY-SERVER")
        DB_NAME     — Database name (e.g. "MyDatabase")
        DB_UID      — SQL Server login username
        DB_PASSWORD — SQL Server login password
    """
    server   = os.getenv("DB_SERVER", "localhost")
    database = os.getenv("DB_NAME", "RAGDatabase")
    uid      = os.getenv("DB_UID")
    pwd      = os.getenv("DB_PASSWORD")

    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={uid};"
        f"PWD={pwd};"
        "Encrypt=optional;"
        "TrustServerCertificate=yes;"
    )

    try:
        conn = pyodbc.connect(conn_str)
        print("Database connection successful!")
        return conn
    except Exception as e:
        print("Connection error:", e)
        raise
