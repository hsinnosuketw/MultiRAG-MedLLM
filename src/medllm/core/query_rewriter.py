from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .prompt import VectorstoreQueryRewriterPrompt_W_NER, GraphRAGQueryRewriterPrompt, TabularRAGQueryRewriterPrompt
from src.medllm.config.config import MODEL_ID_VS, MODEL_ID_GRAPH, MODEL_ID_TAB

# vectorstore
def get_vectorstore_llm():
    return ChatNVIDIA(model=MODEL_ID_VS, temperature=0)

def rewrite_vectorstore_query(question, drug_tag_list, drug_list_extracted):
    llm = get_vectorstore_llm()
    if drug_list_extracted:
        prompt = PromptTemplate(
            template=VectorstoreQueryRewriterPrompt_W_NER,
            input_variables=["question", "drug_tag_list", "d_list_ext"]
        )
        query_rewriter = prompt | llm | StrOutputParser()
        return query_rewriter.invoke({"question": question, "drug_tag_list": drug_tag_list, "d_list_ext": drug_list_extracted})
    return ""

# graphRAG
def get_graph_llm():
    return ChatNVIDIA(model=MODEL_ID_GRAPH, temperature=0)

def rewrite_graph_query(question):
    prompt = PromptTemplate(template=GraphRAGQueryRewriterPrompt, input_variables=["question"])
    llm = get_graph_llm()
    query_rewriter = prompt | llm | StrOutputParser()
    return query_rewriter.invoke({"question": question})

# tabularRAG
def get_tabular_llm():
    return ChatNVIDIA(model=MODEL_ID_TAB, temperature=0)

def rewrite_tabular_query(question):
    prompt = PromptTemplate(template=TabularRAGQueryRewriterPrompt, input_variables=["question"])
    llm = get_tabular_llm()
    query_rewriter = prompt | llm | StrOutputParser()
    return query_rewriter.invoke({"question": question}).replace("\n", "").replace("cpilevel", "cpiclevel")
