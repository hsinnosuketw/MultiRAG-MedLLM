# Langchain react agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.medllm.config.config import MODEL_ID_RETRIEVER
from src.medllm.utils.tool_functions import query_cpic, query_interaction
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from src.medllm.core.prompt import RetrieverPrompt

def retrieve(question):
    # Create the agent
    memory = MemorySaver()
    model = ChatNVIDIA(model=MODEL_ID_RETRIEVER, temperature=0)
    tools = [query_cpic, query_interaction]
    retriever = create_react_agent(model, tools, checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 10}
        
    retrieving = retriever.invoke(
        {"messages": [SystemMessage(content=RetrieverPrompt), HumanMessage(content=question)]},
        config,
        stream_mode="values",
    )
    return retrieving['messages'][-1].content

def retrieve_from_vectorstore(vectorstore, query, drug_list, k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = []
    for drug in drug_list:
        result = vectorstore.similarity_search(query, filter={"drug_name": drug}, k=k if len(drug_list) <= 3 else 1)
        results.extend(result)
    return results