from src.medllm.core.vectorstore import retrieve_from_chroma
from src.medllm.core.retriever import retrieve
from src.medllm.core.grader import rank_documents, generate_answer
from src.medllm.utils.utils import measure_time # load_list_from_file
from langchain_core.documents import Document
from drug_named_entity_recognition import find_drugs
import sys


# @measure_time
def main(question):
    # 將問題中的藥物名稱(可能為商品名)統一改為資料庫中的名稱
    drug_list_extracted = find_drugs(question.split(" "))
    for drug in drug_list_extracted:
        question = question.replace(drug[0]["matching_string"], drug[0]["name"])

    # 檢索資訊
    vs_docs = retrieve_from_chroma(question, n_results=5)
    retrieved_info = retrieve(question)

    # 將檢索倒的資訊轉換為 Document 格式
    docs = vs_docs

    retrieved_docs = Document(page_content=str(retrieved_info))
    docs.extend([retrieved_docs])

    # 排序並過濾檢索到的資訊
    ranked_docs = rank_documents(question, docs)

    # 生成回應
    final_answer = generate_answer(question, ranked_docs) # TODO: 修改 Answer generator 的 prompt 使其可以將沒找到對應的互動理解成 "資料庫中沒有相關互動，請諮詢醫護人員"

    return final_answer

if __name__ == "__main__":
    print(main(sys.argv[1]))