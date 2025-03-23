from neo4j import GraphDatabase
from ..config.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, SQLITE_DB_PATH
import sqlite3

def retrieve_from_vectorstore(vectorstore, query, drug_list, k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = []
    for drug in drug_list:
        result = vectorstore.similarity_search(query, filter={"drug_name": drug}, k=k if len(drug_list) <= 3 else 1)
        results.extend(result)
    return results

def query_graph(graph_query):
    if not graph_query:
        return ""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()  # 驗證連線
        with driver.session() as session:
            result = session.run(graph_query)
            res = [record["r"]["description"] for record in result if "r" in record] or [""]
            return res[0]
    except Exception as e:
        print(f"GraphRAG query failed: {str(e)}")
        return ""
    finally:
        driver.close()

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