from langchain_community.vectorstores import Chroma
from .embeddings import get_embedding_function
from ..config.config import PERSIST_DIRECTORY, COLLECTION_NAME

def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=PERSIST_DIRECTORY
    )

def retrieve_from_vectorstore(vectorstore, query, drug_list, k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = []
    for drug in drug_list:
        result = vectorstore.similarity_search(query, filter={"drug_name": drug}, k=k if len(drug_list) <= 3 else 1)
        results.extend(result)
    return results