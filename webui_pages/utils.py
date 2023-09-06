from typing import *
from pathlib import Path
from configs import (
    LLM_MODEL,
    logger,
)
import httpx
import asyncio
from server.chat.openai_chat import OpenAiChatMsgIn
from fastapi.responses import StreamingResponse
import contextlib
import json
from io import BytesIO
from server.utils import run_async, iter_over_async

from configs import NLTK_DATA_PATH
import nltk
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


def set_httpx_timeout(timeout=60.0):
    '''
    httpx default timeout 5s
    '''
    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout


set_httpx_timeout()


class ApiRequest:
    '''
    api.py
    1. send request to api
    2. call function directly without api
    '''
    def __init__(
        self,
        # base_url: str = "http://127.0.0.1:7861",
        llm_url: str,
        timeout: float = 60.0,
        no_remote_api: bool = False,   # call api view function directly
    ):
        self.base_url = ""
        self.llm_url = llm_url
        self.timeout = timeout
        self.no_remote_api = no_remote_api

    def _parse_url(self, service: str, url: str) -> str:
        if service == "kb_service":
            self.base_url = self.kb_url
        elif service == "llm_service":
            self.base_url = self.llm_url
        if (not url.startswith("http")
                    and self.base_url
                ):
            part1 = self.base_url.strip(" /")
            part2 = url.strip(" /")
            return f"{part1}/{part2}"
        else:
            return url

    def get(
        self,
        service: str,
        url: str,
        params: Union[Dict, List[Tuple], bytes] = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(service, url)
        kwargs.setdefault("timeout", self.timeout)
        while retry > 0:
            try:
                if stream:
                    return httpx.stream("GET", url, params=params, **kwargs)
                else:
                    return httpx.get(url, params=params, **kwargs)
            except Exception as e:
                logger.error(e)
                retry -= 1

    async def aget(
        self,
        url: str,
        params: Union[Dict, List[Tuple], bytes] = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(url)
        kwargs.setdefault("timeout", self.timeout)
        async with httpx.AsyncClient() as client:
            while retry > 0:
                try:
                    if stream:
                        return await client.stream("GET", url, params=params, **kwargs)
                    else:
                        return await client.get(url, params=params, **kwargs)
                except Exception as e:
                    logger.error(e)
                    retry -= 1

    def post(
        self,
        svc: str,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(service=svc, url=url)
        kwargs.setdefault("timeout", self.timeout)
        while retry > 0:
            try:
                # return requests.post(url, data=data, json=json, stream=stream, **kwargs)
                if stream:
                    return httpx.stream("POST", url, data=data, json=json, **kwargs)
                else:
                    return httpx.post(url, data=data, json=json, **kwargs)
            except Exception as e:
                logger.error(e)
                retry -= 1

    async def apost(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(url)
        kwargs.setdefault("timeout", self.timeout)
        async with httpx.AsyncClient() as client:
            while retry > 0:
                try:
                    if stream:
                        return await client.stream("POST", url, data=data, json=json, **kwargs)
                    else:
                        return await client.post(url, data=data, json=json, **kwargs)
                except Exception as e:
                    logger.error(e)
                    retry -= 1

    def delete(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(url)
        kwargs.setdefault("timeout", self.timeout)
        while retry > 0:
            try:
                if stream:
                    return httpx.stream("DELETE", url, data=data, json=json, **kwargs)
                else:
                    return httpx.delete(url, data=data, json=json, **kwargs)
            except Exception as e:
                logger.error(e)
                retry -= 1

    async def adelete(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[httpx.Response, None]:
        url = self._parse_url(url)
        kwargs.setdefault("timeout", self.timeout)
        async with httpx.AsyncClient() as client:
            while retry > 0:
                try:
                    if stream:
                        return await client.stream("DELETE", url, data=data, json=json, **kwargs)
                    else:
                        return await client.delete(url, data=data, json=json, **kwargs)
                except Exception as e:
                    logger.error(e)
                    retry -= 1

    def _fastapi_stream2generator(self, response: StreamingResponse, as_json: bool =False):
        '''
        将api.py中视图函数返回的StreamingResponse转化为同步生成器
        '''
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
        
        try:
            for chunk in  iter_over_async(response.body_iterator, loop):
                if as_json and chunk:
                    yield json.loads(chunk)
                elif chunk.strip():
                    yield chunk
        except Exception as e:
            logger.error(e)

    def _httpx_stream2generator(
        self,
        response: contextlib._GeneratorContextManager,
        as_json: bool = False,
    ):
        '''
        将httpx.stream返回的GeneratorContextManager转化为普通生成器
        '''
        try:
            with response as r:
                for chunk in r.iter_text(None):
                    if as_json and chunk:
                        yield json.loads(chunk)
                    elif chunk.strip():
                        yield chunk
        except httpx.ConnectError as e:
            msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。"
            logger.error(msg)
            logger.error(e)
            yield {"code": 500, "msg": msg}
        except httpx.ReadTimeout as e:
            msg = f"API通信超时，请确认已启动FastChat与API服务）"
            logger.error(msg)
            logger.error(e)
            yield {"code": 500, "msg": msg}
        except Exception as e:
            logger.error(e)
            yield {"code": 500, "msg": str(e)}

    # model config request
    def reload_model(
        self,
        model: str,
        model_dtype: str,
        framework: str
        ):
        pass


    # chat 

    def chat_fastchat(
        self,
        messages: List[Dict],
        stream: bool = True,
        model: str = LLM_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        no_remote_api: bool = None,
        **kwargs: Any,
    ):
        '''
        '''
        msg = OpenAiChatMsgIn(**{
            "messages": messages,
            "stream": stream,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        })

        if no_remote_api:
            pass
        else:
            data = msg.dict(exclude_unset=True, exclude_none=True)
            response = self.post(
                "/chat/fastchat",
                json=data,
                stream=stream,
            )
            return self._httpx_stream2generator(response)

    def chat_chat(
        self,
        query: str,
        history: List[Dict] = [],
        stream: bool = True,
        no_remote_api: bool = None,
    ):
        '''
        '''

        data = {
            "query": query,
            "history": history,
            "stream": stream,
        }

        if no_remote_api:
            pass
        else:
            response = self.post("/chat/chat", json=data, stream=True)
            return self._httpx_stream2generator(response)
    
    def list_models(self, service):
        response = self.get(service=service, url="/models")
        if response["status"] == 200:
            return response["models"]
        return MODEL_LIST
    
    def llm_chat(
        self,
        model: str,
        query: str,
        history: List[Dict] = [],
        stream: bool = True
    ):
        data = {
            "prompt": query,
            "history": history
        }
        response = self.post(svc="llm_chat", url="/v1/completions", json=data, stream=False)
        return response

    
    def knowledge_base_chat(
        self,
        question: str,
        selected_kb: str,
        max_token: int = 1000,
        top_k: int = VECTOR_SEARCH_TOP_K,
        top_p: int = 0.8,
        temperature: int = 0.5,
        score_threshold: float = SCORE_THRESHOLD,
        kb_top_k: int = 2,
        with_history: bool = False,
        history: List[Dict] = [],
        stream: bool = True,
    ):
        search_kwargs ={
            "k": kb_top_k
        }
        data = {
            "question": question,
            "file": selected_kb,
            "max_token": max_token,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "score_threshold": score_threshold,
            "search_kwargs": {search_kwargs},
            "with_history": with_history,
            "history": history,
        }

        response = self.post(
            "/query",
            json=data
        )
        return response
        
    def search_engine_chat(
        self,
        query: str,
        search_engine_name: str,
        top_k: int = SEARCH_ENGINE_TOP_K,
        stream: bool = True,
        no_remote_api: bool = None,
    ):

        data = {
            "query": query,
            "search_engine_name": search_engine_name,
            "top_k": top_k,
            "stream": stream,
        }

        if no_remote_api:
            pass
        else:
            response = self.post(
                "/chat/search_engine_chat",
                json=data,
                stream=True,
            )
            return self._httpx_stream2generator(response, as_json=True)

    # knowledge base

    def _check_httpx_json_response(
            self,
            response: httpx.Response,
            errorMsg: str = f"failed to connect API server",
        ) -> Dict:
        '''
        check whether httpx returns correct data with normal Response.
        error in api with streaming support was checked in _httpx_stream2enerator
        '''
        try:
            return response.json()
        except Exception as e:
            logger.error(e)
            return {"code": 500, "msg": errorMsg or str(e)}

    def list_knowledge_bases(
        self,
        no_remote_api: bool = None,
    ):
        '''
        '''
        if no_remote_api:
            response = self.get("/knowledge_base/list_knowledge_bases")
            data = self._check_httpx_json_response(response)
            return data.get("data", [])

    def create_knowledge_base(
        self,
        knowledge_base_name: str,
        vector_store_type: str = "faiss",
        embed_model: str = EMBEDDING_MODEL,
        no_remote_api: bool = None,
    ):

        data = {
            "knowledge_base_name": knowledge_base_name,
            "vector_store_type": vector_store_type,
            "embed_model": embed_model,
        }

        if no_remote_api:
            pass
        else:
            response = self.post(
                "/knowledge_base/create_knowledge_base",
                json=data,
            )
            return self._check_httpx_json_response(response)

    def delete_knowledge_base(
        self,
        knowledge_base_name: str,
        no_remote_api: bool = None,
    ):
        '''
        '''

        if no_remote_api:
            pass
        else:
            response = self.post(
                "/knowledge_base/delete_knowledge_base",
                json=f"{knowledge_base_name}",
            )
            return self._check_httpx_json_response(response)

    def list_kb_docs(
        self,
        knowledge_base_name: str,
        no_remote_api: bool = None,
    ):
        '''
        '''

        if no_remote_api:
            pass
        else:
            response = self.get(
                "/knowledge_base/list_docs",
                params={"knowledge_base_name": knowledge_base_name}
            )
            data = self._check_httpx_json_response(response)
            return data.get("data", [])

    def upload_kb_doc(
        self,
        file: Union[str, Path, bytes],
        knowledge_base_name: str,
        filename: str = None,
        override: bool = False,
        not_refresh_vs_cache: bool = False,
        no_remote_api: bool = None,
    ):
        '''
        '''
        if isinstance(file, bytes): # raw bytes
            file = BytesIO(file)
        elif hasattr(file, "read"): # a file io like object
            filename = filename or file.name
        else: # a local path
            file = Path(file).absolute().open("rb")
            filename = filename or file.name

        if no_remote_api:
            return

    def delete_kb_doc(
        self,
        knowledge_base_name: str,
        doc_name: str,
        delete_content: bool = False,
        not_refresh_vs_cache: bool = False,
        no_remote_api: bool = None,
    ):
        '''
        '''

        data = {
            "knowledge_base_name": knowledge_base_name,
            "doc_name": doc_name,
            "delete_content": delete_content,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        if no_remote_api:
            pass
        else:
            response = self.post(
                "/knowledge_base/delete_doc",
                json=data,
            )
            return self._check_httpx_json_response(response)

    def update_kb_doc(
        self,
        knowledge_base_name: str,
        file_name: str,
        not_refresh_vs_cache: bool = False,
        no_remote_api: bool = None,
    ):
        '''
        '''

        if no_remote_api:
            pass
        else:
            response = self.post(
                "/knowledge_base/update_doc",
                json={
                    "knowledge_base_name": knowledge_base_name,
                    "file_name": file_name,
                    "not_refresh_vs_cache": not_refresh_vs_cache,
                },
            )
            return self._check_httpx_json_response(response)

    def recreate_vector_store(
        self,
        knowledge_base_name: str,
        allow_empty_kb: bool = True,
        vs_type: str = DEFAULT_VS_TYPE,
        embed_model: str = EMBEDDING_MODEL,
        no_remote_api: bool = None,
    ):
        '''
        '''

        data = {
            "knowledge_base_name": knowledge_base_name,
            "allow_empty_kb": allow_empty_kb,
            "vs_type": vs_type,
            "embed_model": embed_model,
        }

        if no_remote_api:
            pass
        else:
            response = self.post(
                "/knowledge_base/recreate_vector_store",
                json=data,
                stream=True,
                timeout=None,
            )
            return self._httpx_stream2generator(response, as_json=True)


def check_error_msg(data: Union[str, dict, list], key: str = "errorMsg") -> str:
    '''
    return error message if error occured when requests API
    '''
    if isinstance(data, dict):
        if key in data:
            return data[key]
        if "code" in data and data["code"] != 200:
            return data["msg"]
    return ""


def check_success_msg(data: Union[str, dict, list], key: str = "msg") -> str:
    '''
    return error message if error occured when requests API
    '''
    if (isinstance(data, dict)
        and key in data
        and "code" in data
        and data["code"] == 200):
        return data[key]
    return ""


if __name__ == "__main__":
    api = ApiRequest(no_remote_api=True)
