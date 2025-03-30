from src.medllm.core.vectorstore import retrieve_from_chroma
from src.medllm.core.retriever import retrieve
from src.medllm.core.grader import grade_retrieval, filter_retrieval, rank_documents, generate_answer
from src.medllm.utils.utils import measure_time # load_list_from_file
from langchain_core.documents import Document
from drug_named_entity_recognition import find_drugs



@measure_time
def main(question):
    # 將問題中的藥物名稱(可能為商品名)統一改為資料庫中的名稱
    drug_list_extracted = find_drugs(question.split(" "))
    for drug in drug_list_extracted:
        question = question.replace(drug[0]["matching_string"], drug[0]["name"])
    # print("Rewrited question: " + question)

    vs_docs = retrieve_from_chroma(question, n_results=5)
    retrieved_info = retrieve(question)

    # 準備文檔
    docs = vs_docs
    retrieved_docs = Document(page_content=str(retrieved_info))
    docs.extend([retrieved_docs])
    print()
    print(docs)
    print()

    # 評分與過濾
    graded_docs = grade_retrieval(question, docs)
    print(graded_docs)
    filtered_docs = filter_retrieval(question, graded_docs)

    # 排序與生成回答
    ranked_docs = rank_documents(question, filtered_docs)
    final_answer = generate_answer(question, ranked_docs) # TODO: 修改 Answer generator 的 prompt 使其可以將沒找到對應的互動理解成 "資料庫中沒有相關互動，請諮詢醫護人員"

    return final_answer

if __name__ == "__main__":
    question = "Can you use Winlevi and tretinoin together?"
    print(main(question))