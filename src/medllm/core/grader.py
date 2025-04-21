from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.medllm.core.prompt import AnswerGenerationPrompt
from src.medllm.config.config import MODEL_ID_RETRIEVER, API_KEY

def get_retrieval_llm():
    """
    Initialize the NVIDIA retrieval model.

    This function sets the temperature to 0 for deterministic results.

    Returns:
        ChatNVIDIA: An instance of the ChatNVIDIA class configured for retrieval.
    """
    return ChatNVIDIA(model=MODEL_ID_RETRIEVER, temperature=0)

def rank_documents(question, documents):
    """
    Rank documents using NVIDIA's reranking model.

    Args:
        question (str): The query string.
        documents (list): List of Document objects to be ranked.

    Returns:
        list: List of Document objects sorted by relevance score.
    """
    ranker = NVIDIARerank(model="nv-rerank-qa-mistral-4b:1", api_key=API_KEY)
    ranker.top_n = 5
    threshold = 0
    reranked_docs = ranker.compress_documents(query=question, documents=documents)
    return [doc for doc in reranked_docs if doc.metadata.get("relevance_score", 0) > threshold]

def generate_answer(question, context):
    """
    Generate an answer to a question using answer generator.

    Args:
        question (str): The question to be answered.
        context (list): List of Document objects containing the context.

    Returns:
        str: The generated answer.
    """

    prompt = PromptTemplate(template=AnswerGenerationPrompt, input_variables=["question", "context"])
    llm = get_retrieval_llm()
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": context, "question": question})
    generation = {"answer" : generation}
    return generation['answer']