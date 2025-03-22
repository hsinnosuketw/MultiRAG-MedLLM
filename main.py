from src.medllm.core.vectorstore import get_vectorstore, retrieve_from_vectorstore
from src.medllm.core.graphRag import rewrite_graph_query, query_graph
from src.medllm.core.tabularRag import rewrite_tabular_query, query_tabular
from src.medllm.core.query_rewriter import rewrite_vectorstore_query
from src.medllm.core.retriever import grade_retrieval, filter_retrieval, rank_documents, generate_answer
from src.medllm.utils.utils import measure_time, load_list_from_file
from langchain_core.documents import Document
import src.medllm.utils.extract as extract

@measure_time
def main(question):
    # 載入藥物和標籤清單
    drug_tag_list = load_list_from_file('src/medllm/utils/drugTagList.txt')
    drug_list = load_list_from_file('src/medllm/utils/drugName.txt')

    # Vectorstore 重寫與檢索
    drug_list_extracted = extract.extract_drug(question)
    query_vs = rewrite_vectorstore_query(question, drug_tag_list, drug_list_extracted)
    vectorstore = get_vectorstore()
    vs_docs = retrieve_from_vectorstore(vectorstore, query_vs, drug_list_extracted) if drug_list_extracted else []

    # GraphRAG 重寫與檢索
    graph_query = rewrite_graph_query(question)
    graph_result = query_graph(graph_query) if graph_query else ""

    # TabularRAG 重寫與檢索
    sql_query = rewrite_tabular_query(question)
    sql_result = query_tabular(sql_query) if sql_query else ""

    # 準備文檔
    docs = vs_docs
    tab_doc = Document(page_content=str(sql_result).replace("'", "") + (sql_query.split("|", 1)[1] if len(sql_query.split("|", 1)) > 1 else ""),
                      metadata={"source": "TabularRAG"})
    graph_doc = Document(page_content=graph_result, metadata={"source": "graphRAG"})
    docs.extend([tab_doc, graph_doc])

    # 評分與過濾
    graded_docs = grade_retrieval(question, docs)
    filtered_docs = filter_retrieval(question, docs)

    # 排序與生成回答
    ranked_docs = rank_documents(question, filtered_docs)
    final_answer = generate_answer(question, ranked_docs)

    return final_answer

if __name__ == "__main__":
    question = "Opdivo vs Opdivo Qvantig: What is the Difference?"
    print(main(question))