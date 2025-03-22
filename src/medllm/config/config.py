import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

# 配置參數
API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_ID_VS = "meta/llama-3.3-70b-instruct"
MODEL_ID_GRAPH = "meta/llama-3.3-70b-instruct"
MODEL_ID_TAB = "meta/llama-3.3-70b-instruct"
MODEL_ID_FILTER = "meta/llama-3.1-405b-instruct"
MODEL_NAME_EMBED = "all-MiniLM-L6-v2.gguf2.f16.gguf"
GPT4ALL_KWARGS = {'allow_download': 'True'}
PERSIST_DIRECTORY = "./Trial_chroma_langchain"
COLLECTION_NAME = "Trial_v1"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
SQLITE_DB_PATH = "./drug.db"

