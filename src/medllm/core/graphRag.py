from neo4j import GraphDatabase
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .prompt import GraphRAGQueryRewriterPrompt
from ..config.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, MODEL_ID_GRAPH

def get_graph_llm():
    return ChatNVIDIA(model=MODEL_ID_GRAPH, temperature=0)

def rewrite_graph_query(question):
    prompt = PromptTemplate(template=GraphRAGQueryRewriterPrompt, input_variables=["question"])
    llm = get_graph_llm()
    query_rewriter = prompt | llm | StrOutputParser()
    return query_rewriter.invoke({"question": question})

# def query_graph(graph_query):
#     driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
#     print(driver)
#     with driver.session() as session:
#         result = session.run(graph_query)
#         res = [record["r"]["description"] for record in result] if result else []
#     driver.close()
#     return res[0] if res else ""
def query_graph(query):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run(query)
        res = []
        for record in result:
            res.append(record)
        return res
# from neo4j import GraphDatabase
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from prompt import GraphRAGQueryRewriterPrompt
# from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, MODEL_ID_GRAPH

def get_graph_llm():
    return ChatNVIDIA(model=MODEL_ID_GRAPH, temperature=0)

def rewrite_graph_query(question):
    prompt = PromptTemplate(template=GraphRAGQueryRewriterPrompt, input_variables=["question"])
    llm = get_graph_llm()
    query_rewriter = prompt | llm | StrOutputParser()
    return query_rewriter.invoke({"question": question})

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