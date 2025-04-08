import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import sys
import subprocess

# 檢查是否有 CUDA GPU 可用
if not torch.cuda.is_available():
    # print("No CUDA GPU available. Switching to CPU.")
    device = torch.device('cpu')
else:
    def get_nvidia_smi_memory():
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'],
                capture_output=True, text=True, check=True
            )
            lines = result.stdout.strip().split('\n')
            gpu_memory = {}
            for i, line in enumerate(lines):
                total, used = map(int, line.split(', '))
                gpu_memory[i] = {'total': total, 'used': used, 'free': total - used}
            return gpu_memory
        except subprocess.CalledProcessError as e:
            # print(f"Error running nvidia-smi: {e}")
            return None

    def select_gpu_with_enough_memory(min_required_vram_mb=13000):
        memory_info = get_nvidia_smi_memory()
        if memory_info is None:
            return torch.device('cpu')

        for gpu_id in range(torch.cuda.device_count()):
            if gpu_id in memory_info:
                total_memory = memory_info[gpu_id]['total']
                free_memory = memory_info[gpu_id]['free']

                # print(f"GPU {gpu_id}:")
                # print(f"  Total Memory: {total_memory:.2f} MB")
                # print(f"  Used Memory: {memory_info[gpu_id]['used']:.2f} MB")
                # print(f"  Free Memory: {free_memory:.2f} MB")

                if free_memory >= min_required_vram_mb:
                    # print(f"GPU {gpu_id} has sufficient memory ({free_memory:.2f} MB >= {min_required_vram_mb} MB).")
                    return torch.device(f'cuda:{gpu_id}')
        # print("No GPU has sufficient memory. Switching to CPU.")
        return torch.device('cpu')

    # 設定模型所需的預估 VRAM
    required_vram_mb = 13000  # 約 13GB
    device = select_gpu_with_enough_memory(min_required_vram_mb=required_vram_mb)
    # print(f"Selected device: {device}")

# 全域變數，將模型和 tokenizer 載入到選定設備
tokenizer = AutoTokenizer.from_pretrained('infly/inf-retriever-v1-1.5b', trust_remote_code=True)
model = AutoModel.from_pretrained('infly/inf-retriever-v1-1.5b', trust_remote_code=True).to(device)
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
    if task:
        input_text = f'Instruct: {task}\nQuery: {sentence}'
    else:
        input_text = sentence

    batch_dict = tokenizer([input_text], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0]

if __name__ == "__main__":
    test_sentence = "This is a test sentence."
    embedding_vector = embedding(test_sentence)
    # print(f"Embedding shape: {embedding_vector.shape}")
    # print(f"Embedding device: {embedding_vector.device}")

# from langchain_community.embeddings import GPT4AllEmbeddings
# from ..config.config import MODEL_NAME_EMBED, GPT4ALL_KWARGS

# # def get_embedding_function():
# #     return GPT4AllEmbeddings(
# #         model_name=MODEL_NAME_EMBED,
# #         gpt4all_kwargs=GPT4ALL_KWARGS,
# #         # device="cuda",
# #     )

# # for new vectorstore

# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from transformers import AutoTokenizer, AutoModel

# # 檢查是否有 CUDA GPU 可用
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # print(f"Using device: {device}")

# # 全域變數，將模型和 tokenizer 載入到 GPU
# tokenizer = AutoTokenizer.from_pretrained('infly/inf-retriever-v1-1.5b', trust_remote_code=True)
# model = AutoModel.from_pretrained('infly/inf-retriever-v1-1.5b', trust_remote_code=True).to(device)  # 移動到 GPU
# max_length = 8192

# def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
#     left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#     if left_padding:
#         return last_hidden_states[:, -1]
#     else:
#         sequence_lengths = attention_mask.sum(dim=1) - 1
#         batch_size = last_hidden_states.shape[0]
#         return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# def embedding(sentence: str, task: str = None) -> Tensor:
#     """
#     將單一句子轉換為嵌入向量，使用 GPU 加速。
#     :param sentence: 輸入句子
#     :param task: 可選的任務描述，若提供則生成帶指令的查詢嵌入
#     :return: 正規化後的嵌入向量 (Tensor)
#     """
#     if task:
#         input_text = f'Instruct: {task}\nQuery: {sentence}'
#     else:
#         input_text = sentence

#     # 將文本轉換為 token，移動到 GPU
#     # # print(device)
#     batch_dict = tokenizer([input_text], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
#     batch_dict = {k: v.to(device) for k, v in batch_dict.items()}  # 將所有張量移到 GPU
    
#     # 生成嵌入
#     with torch.no_grad():  # 禁用梯度計算以節省記憶體
#         outputs = model(**batch_dict)
#         embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
#     # 正規化嵌入
#     embeddings = F.normalize(embeddings, p=2, dim=1)
#     return embeddings[0]  # 返回單個嵌入向量，仍在 GPU 上