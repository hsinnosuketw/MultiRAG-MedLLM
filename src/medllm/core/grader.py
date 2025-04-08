from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document
from src.medllm.core.prompt import RetrieverFilterPrompt, AnswerGenerationPrompt
from src.medllm.config.config import MODEL_ID_FILTER, API_KEY

def get_retrieval_llm():
    return ChatNVIDIA(model=MODEL_ID_FILTER, temperature=0)

def grade_retrieval(question, documents):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question or is a deterministic answer to the user question,
        grade it as relevant. Give a binary score "yes" or "no" in JSON with key "score".
        EXAMPLE: {"score" : "yes"} or {"score" : "no"}
        <|eot_id|><|start_header_id|>user<|end_header_id>
        Document: {document}\nQuestion: {question} <|eot_id|>""",
        input_variables=["question", "document"]
    )
    llm = get_retrieval_llm()
    retrieval_grader = prompt | llm | JsonOutputParser()
    return retrieval_grader.invoke({"question": question, "document": documents})

def filter_retrieval(question, documents):
    prompt = PromptTemplate(template=RetrieverFilterPrompt, input_variables=["question", "documents"])
    llm = get_retrieval_llm()
    retrieval_filter = prompt | llm | JsonOutputParser()
    filtered = retrieval_filter.invoke({"question": question, "documents": documents})
    return [Document(page_content=f["page_content"]) for f in filtered['filtered docs']]

def rank_documents(question, documents):
    ranker = NVIDIARerank(model="nv-rerank-qa-mistral-4b:1", api_key=API_KEY)
    ranker.top_n = 5
    threshold = 0
    reranked_docs = ranker.compress_documents(query=question, documents=documents)
    return [doc for doc in reranked_docs if doc.metadata.get("relevance_score", 0) > threshold]

def generate_answer(question, context):
    prompt = PromptTemplate(template=AnswerGenerationPrompt, input_variables=["question", "context"])
    llm = get_retrieval_llm()
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": context, "question": question})
    generation = {"answer" : generation}
    return generation['answer']