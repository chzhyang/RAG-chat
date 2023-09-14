import os
import logging
# import torch
# log format
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

# Service config
LLM_SERVICE_URL = "http://127.0.0.1:8000"
KB_SERVICE_URL = "http://127.0.0.1:12000"

# Model config
MODEL_CONFIG = {
    "model": "chatglm2-6b",
    "datatype": "INT4",
    "framework": "bigdl-llm",
    "temperature": 0.9,
    "top_p": 0.9,
    "max_length": 128,
    "history_len": 0,
}
LLM_DEVICE="cpu"
# LLM_DEVICE = "cuda" if torch.cuda.is_available(
# ) else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_LIST = ["chatglm2-6b", "llama2-7b"]
MODEL_DTYPE_LIST = ["INT4", "FP16"]
FRAMEWORK_LIST = ["bigdl-llm", "transformers"]


llm_model_dict = {
    "chatglm-6b": {
        # "local_model_path": "THUDM/chatglm-6b",
        "local_model_path": "/home/sdp/models/chatglm2-6b",
        "api_base_url": "http://localhost:12000/v1",
        "api_key": "EMPTY"
    },

    "chatglm-6b-int4": {
        "local_model_path": "THUDM/chatglm-6b-int4",
        "api_base_url": "http://localhost:8888/v1",
        "api_key": "EMPTY"
    },
}

# Knowledge base config
KB_DICT = {
    "kb_sample": {
        "kb_name": "sample1",
        "vs_type": "milvus",
        "files": ["fake.txt"], # file name, TODO: add file path
        "embed_model": "m3e-base"
    },   
    "kb_sample2": {
        "kb_name": "sample2",
        "vs_type": "milvus",
        "files": [],
        "embed_model": ""
    }
}
# If local run, use host path
# If run in contianer, use container path
KB_ROOT_PATH = "/home/sdp/cloud.performance.generative.ai.workload/langchain_demo/files"

KB_VS_DICT = {
    "faiss": {
    },
    "milvus": {
        "host": "127.0.0.1",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": False,
    },
}
DEFAULT_VS_TYPE = "milvus"

CACHED_VS_NUM = 1

CHUNK_SIZE = 250

OVERLAP_SIZE = 50

VECTOR_SEARCH_TOP_K = 5

SCORE_THRESHOLD = 1

# Use local model: "text2vec": "User/Downloads/text2vec-large-chinese" (absolutely path)
EMBEDDING_MODEL_DICT = {
    # "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    # "ernie-base": "nghuyong/ernie-3.0-base-zh",
    # "text2vec-base": "shibing624/text2vec-base-chinese",
    # "text2vec": "GanymedeNil/text2vec-large-chinese",
    # "text2vec-paraphrase": "shibing624/text2vec-base-chinese-paraphrase",
    # "text2vec-sentence": "shibing624/text2vec-base-chinese-sentence",
    "text2vec-multilingual": "shibing624/text2vec-base-multilingual",
    # "m3e-small": "moka-ai/m3e-small",
    # "m3e-base": "moka-ai/m3e-base",
    "m3e-base": "/home/sdp/models/m3e-base",
    # "m3e-large": "moka-ai/m3e-large",
    # "bge-small-zh": "BAAI/bge-small-zh",
    # "bge-base-zh": "BAAI/bge-base-zh",
    # "bge-large-zh": "BAAI/bge-large-zh"
}

EMBEDDING_MODEL = "m3e-base"
# EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
# ) else "mps" if torch.backends.mps.is_available() else "cpu"
EMBEDDING_DEVICE = "cpu"
LOADER_DICT_V1 = {
    # "UnstructuredHTMLLoader": ['.html'],
    # "UnstructuredMarkdownLoader": ['.md'],
    # "CustomJSONLoader": [".json"],
    # "CSVLoader": [".csv"],
    "RapidOCRPDFLoader": [".pdf"],
    # "RapidOCRLoader": ['.png', '.jpg', '.jpeg', '.bmp'],
    # "UnstructuredFileLoader": ['.eml', '.msg', '.rst',
    #                             '.rtf', '.txt', '.xml',
    #                             '.doc', '.docx', '.epub', '.odt',
    #                             '.ppt', '.pptx', '.tsv'],  # '.xlsx'
}
# Log config
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
