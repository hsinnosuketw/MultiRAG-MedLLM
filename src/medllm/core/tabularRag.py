import sqlite3
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .prompt import TabularRAGQueryRewriterPrompt
from ..config.config import SQLITE_DB_PATH, MODEL_ID_TAB

def get_tabular_llm():
    return ChatNVIDIA(model=MODEL_ID_TAB, temperature=0)

def rewrite_tabular_query(question):
    prompt = PromptTemplate(template=TabularRAGQueryRewriterPrompt, input_variables=["question"])
    llm = get_tabular_llm()
    query_rewriter = prompt | llm | StrOutputParser()
    return query_rewriter.invoke({"question": question}).replace("\n", "").replace("cpilevel", "cpiclevel")

def query_tabular(sql_query):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query.split("|")[0].strip())
        result = cursor.fetchall()
    except Exception:
        result = ""
    cursor.close()
    conn.close()
    return result