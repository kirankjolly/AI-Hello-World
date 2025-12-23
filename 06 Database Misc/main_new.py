import sqlite3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import re

# IN THIS CODE THERE ARE NO TOOLS USED. JUST DIRECT LLM CALLS.

# ---------------- ENV ----------------
load_dotenv()
os.makedirs("reports", exist_ok=True)

# ---------------- DB ----------------
conn = sqlite3.connect("db.sqlite")


def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in c.fetchall()]


def describe_tables(table_names):
    c = conn.cursor()
    tables = ", ".join(f"'{t}'" for t in table_names)
    rows = c.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables})"
    )
    return "\n".join(row[0] for row in rows if row[0])


def run_sql(query: str):
    c = conn.cursor()
    c.execute(query)
    return c.fetchall()


# ---------------- LLM ----------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ---------------- STEP 1: Generate SQL ----------------
def generate_sql(user_question: str) -> str:
    tables = list_tables()
    schemas = describe_tables(tables)

    messages = [
        SystemMessage(
            content=(
                "You are a senior data engineer.\n"
                "Write ONLY a valid SQLite SELECT query.\n"
                "Do not guess columns.\n"
                "Do not explain anything.\n\n"
                f"Tables:\n{schemas}"
            )
        ),
        HumanMessage(content=user_question)
    ]

    sql = llm.invoke(messages).content.strip()

    sql = sql.strip()

    # Remove markdown fences if present
    sql = re.sub(r"^```sql|```$", "", sql, flags=re.IGNORECASE).strip()

    # Extract first SELECT statement
    match = re.search(r"(select\s+.*)", sql, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Unsafe SQL generated")

    sql = match.group(1)

    return sql


# ---------------- STEP 2: Generate HTML Report ----------------
def generate_html_report(data, title: str) -> str:
    messages = [
        SystemMessage(
            content=(
                "Generate a clean HTML report.\n"
                "Use a table.\n"
                "Add a title.\n"
                "No markdown.\n"
                "No explanation."
            )
        ),
        HumanMessage(
            content=f"Title: {title}\nData:\n{data}"
        )
    ]

    return llm.invoke(messages).content


# ---------------- STEP 3: Save Report ----------------
def write_report(filename: str, html: str):
    path = os.path.join("reports", filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Report written to {path}")


# ---------------- MAIN WORKFLOW ----------------
def run_workflow(question: str, report_name: str):
    print("\n🔍 Generating SQL...")
    sql = generate_sql(question)
    print(sql)

    print("\n📊 Running SQL...")
    result = run_sql(sql)

    print("\n📝 Generating HTML report...")
    html = generate_html_report(result, question)

    write_report(report_name, html)


# ---------------- RUN ----------------
if __name__ == "__main__":

    run_workflow(
        "How many orders are there?",
        "orders_report.html"
    )

    run_workflow(
        "How many users are there?",
        "users_report.html"
    )
