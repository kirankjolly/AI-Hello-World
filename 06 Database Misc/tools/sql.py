import sqlite3
from langchain.tools import Tool

conn = sqlite3.connect('db.sqlite')

def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    rows = c.fetchall()
    return ",".join(row[0] for row in rows if row[0] is not None)

def run_sqlite_query(query):
    c = conn.cursor()
    c.execute(query)
    result = c.fetchall()
    return result


run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query",
    func=run_sqlite_query
)

def describe_tables(table_names):
    print(f"Tables: {table_names}")
    c = conn.cursor()
    tables = ', '.join("'" + table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables})")
    return '\n'.join(row[0] for row in rows if row[0] is not None)

describe_table_tool = Tool.from_function(
    name="describe_tables",
    description="Describe tables",
    func=describe_tables
)