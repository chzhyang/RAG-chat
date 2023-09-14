import os
import logging
import torch
# log format
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

# For local embedding model, "text2vec": "/home/text2vec-large-chinese"
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec-paraphrase": "shibing624/text2vec-base-chinese-paraphrase",
    "text2vec-sentence": "shibing624/text2vec-base-chinese-sentence",
    "text2vec-multilingual": "shibing624/text2vec-base-multilingual",
    "m3e-small": "moka-ai/m3e-small",
    # "m3e-base": "moka-ai/m3e-base",
    "m3e-base": "/home/sdp/models/m3e-base",
    "m3e-large": "moka-ai/m3e-large",
    "bge-small-zh": "BAAI/bge-small-zh",
    "bge-base-zh": "BAAI/bge-base-zh",
    "bge-large-zh": "BAAI/bge-large-zh"
}

EMBEDDING_MODEL = "m3e-base"

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

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

    "chatglm2-6b": {
        "local_model_path": "THUDM/chatglm2-6b",
        "api_base_url": "http://localhost:8888/v1",
        "api_key": "EMPTY"
    },

    "chatglm2-6b-32k": {
        "local_model_path": "THUDM/chatglm2-6b-32k",
        "api_base_url": "http://localhost:8888/v1",
        "api_key": "EMPTY"
    },

    "vicuna-13b-hf": {
        "local_model_path": "",
        "api_base_url": "http://localhost:8888/v1",
        "api_key": "EMPTY"
    },

    "gpt-3.5-turbo": {
        "local_model_path": "gpt-3.5-turbo",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": os.environ.get("OPENAI_API_KEY")
    },
}


# model config
MODEL_CONFIG = {
    "model": "chatglm2-6b",
    "datatype": "INT4",
    "framework": "bigdl-llm",
    "top_p": 0.7,
    "temperature": 0.9,
    "max_length": 512,
}
LLM_MODEL="chatglm2-6b"
LLM_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

# knowledge base
KB_LIST = [
    {
        "kb_name": "sample",
        "kb_type": "milvus",
        "file_path": "", # todo: add file path or file name
    }   
]

# KB_ROOT_PATH = os.path.join(os.path.dirname(
#     os.path.dirname(__file__)), "knowledge_base")
KB_ROOT_PATH = "../files"

DB_ROOT_PATH = os.path.join(KB_ROOT_PATH, "info.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_ROOT_PATH}"

KB_VS_CONFIG = {
    "faiss": {
    },
    "milvus": {
        "host": "127.0.0.1",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": False,
    },
    "pg": {
        "connection_uri": "postgresql://postgres:postgres@127.0.0.1:5432/langchain_chatchat",
    }
}

# 默认向量库类型。可选：faiss, milvus, pg.
DEFAULT_VS_TYPE = "faiss"

# 缓存向量库数量
CACHED_VS_NUM = 1

# 知识库中单段文本长度
CHUNK_SIZE = 250

# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50

# 知识库匹配向量数量
VECTOR_SEARCH_TOP_K = 5

# 知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右
SCORE_THRESHOLD = 1

# 搜索引擎匹配结题数量
SEARCH_ENGINE_TOP_K = 5

# nltk 模型存储路径
NLTK_DATA_PATH = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "nltk_data")

# 基于本地知识问答的提示词模版
PROMPT_TEMPLATE = """【指令】根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 

【已知信息】{context} 

【问题】{question}"""

LOADER_DICT = {
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
# API 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = False

# Bing 搜索必备变量
# 使用 Bing 搜索需要使用 Bing Subscription Key,需要在azure port中申请试用bing search
# 具体申请方式请见
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource
# 使用python创建bing api 搜索实例详见:
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
# 注意不是bing Webmaster Tools的api key，

# 此外，如果是在服务器上，报Failed to establish a new connection: [Errno 110] Connection timed out
# 是因为服务器加了防火墙，需要联系管理员加白名单，如果公司的服务器的话，就别想了GG
BING_SUBSCRIPTION_KEY = ""

# 是否开启中文标题加强，以及标题增强的相关配置
# 通过增加标题判断，判断哪些文本为标题，并在metadata中进行标记；
# 然后将文本与往上一级的标题进行拼合，实现文本信息的增强。
ZH_TITLE_ENHANCE = False
