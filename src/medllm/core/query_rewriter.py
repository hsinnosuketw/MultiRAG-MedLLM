from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.medllm.core.prompt import VectorstoreQueryRewriterPrompt, VectorstoreQueryRewriterPrompt_W_NER
from src.medllm.config.config import MODEL_ID_VS

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