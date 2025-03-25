from langchain_community.embeddings import GPT4AllEmbeddings
from ..config.config import MODEL_NAME_EMBED, GPT4ALL_KWARGS

def get_embedding_function():
    return GPT4AllEmbeddings(
        model_name=MODEL_NAME_EMBED,
        gpt4all_kwargs=GPT4ALL_KWARGS,
        # device="cuda",
    )

# for new vectorstore

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# 檢查是否有 CUDA GPU 可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 全域變數，將模型和 tokenizer 載入到 GPU
tokenizer = AutoTokenizer.from_pretrained('infly/inf-retriever-v1-1.5b', trust_remote_code=True)
model = AutoModel.from_pretrained('infly/inf-retriever-v1-1.5b', trust_remote_code=True).to(device)  # 移動到 GPU
max_length = 8192

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def embedding(sentence: str, task: str = None) -> Tensor:
    """
    將單一句子轉換為嵌入向量，使用 GPU 加速。
    :param sentence: 輸入句子
    :param task: 可選的任務描述，若提供則生成帶指令的查詢嵌入
    :return: 正規化後的嵌入向量 (Tensor)
    """
    if task:
        input_text = f'Instruct: {task}\nQuery: {sentence}'
    else:
        input_text = sentence

    # 將文本轉換為 token，移動到 GPU
    # print(device)
    batch_dict = tokenizer([input_text], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}  # 將所有張量移到 GPU
    
    # 生成嵌入
    with torch.no_grad():  # 禁用梯度計算以節省記憶體
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
    # 正規化嵌入
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0]  # 返回單個嵌入向量，仍在 GPU 上