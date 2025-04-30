from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.medllm.core.prompt import AnswerGenerationPrompt, systemPromptV2, systemPromptV5
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
    threshold = -7
    print(f"Reranking {len(documents)} documents")
    try:
        reranked_docs = ranker.compress_documents(query=question, documents=documents)
        # print(f"The most relevant docs:\n{reranked_docs[1].page_content} with score {reranked_docs[1].metadata.get('relevance_score', 0)}")
        # print([doc for doc in reranked_docs if doc.metadata.get("relevance_score", 0) > threshold])
        for doc in reranked_docs:
            print(f" {doc.metadata.get('relevance_score', 0)}", end=", ")
        print()
        return reranked_docs
    except Exception as e:
        print(f"Error during reranking: {e}")
        return documents

def generate_answer(question, context):
    """
    Generate an answer to a question using answer generator.

    Args:
        question (str): The question to be answered.
        context (list): List of Document objects containing the context.

    Returns:
        str: The generated answer.
    """

    prompt = PromptTemplate(template=systemPromptV5, input_variables=["question", "context"])
    llm = get_retrieval_llm()
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": context, "question": question})
    generation = {"answer" : generation}
    return generation['answer']