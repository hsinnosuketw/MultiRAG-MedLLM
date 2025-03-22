from langchain_community.embeddings import GPT4AllEmbeddings
from ..config.config import MODEL_NAME_EMBED, GPT4ALL_KWARGS

def get_embedding_function():
    return GPT4AllEmbeddings(
        model_name=MODEL_NAME_EMBED,
        gpt4all_kwargs=GPT4ALL_KWARGS,
        device="cuda",
    )