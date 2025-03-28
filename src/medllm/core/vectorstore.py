from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from .embeddings import embedding
from ..config.config import PERSIST_DIRECTORY, COLLECTION_NAME

# using old vectorstore
# def get_vectorstore():
#     return Chroma(
#         collection_name=COLLECTION_NAME,
#         embedding_function=get_embedding_function(),
#         persist_directory=PERSIST_DIRECTORY
#     )

# def retrieve_from_vectorstore(vectorstore, query, drug_list, k=5):
#     retriever = vectorstore.as_retriever(search_kwargs={"k": k})
#     results = []
#     for drug in drug_list:
#         result = vectorstore.similarity_search(query, filter={"drug_name": drug}, k=k if len(drug_list) <= 3 else 1)
#         results.extend(result)
#     return results

# using new vectorstore
# 全域 Chroma 客戶端
client = chromadb.PersistentClient(path="./chroma_database_demo")
collection_name = "drugbank"
collection = client.get_or_create_collection(name=collection_name)

def parse_to_langchain_documents(results):
    """
    將 Chroma 檢索結果解析成 LangChain Document 物件列表。
    :param results: Chroma 查詢結果，包含 documents、metadatas 和 distances
    :return: List[Document] - LangChain Document 物件列表
    """
    documents = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        # 創建 LangChain Document 物件
        langchain_doc = Document(
            page_content=doc,  # 文件內容
            metadata={**meta, "distance": dist}  # 合併原始元數據與距離
        )
        documents.append(langchain_doc)
    return documents

def retrieve_from_chroma(query: str, n_results: int = 5) -> dict:
    """
    從 Chroma 資料庫中檢索與查詢最相關的文檔。
    :param query: 查詢文本
    :param n_results: 返回的結果數量
    :return: 包含文檔、距離等的字典
    """
    # 生成查詢嵌入
    query_embedding = embedding(query).cpu().numpy()  # 轉為 numpy 並移回 CPU
    
    # 執行檢索
    results = collection.query(
        query_embeddings=query_embedding.tolist(),  # Chroma 期望列表格式
        n_results=n_results,
        include=["documents", "metadatas", "distances"]  # 指定返回的欄位
    )
    
    return parse_to_langchain_documents(results)