import os
from typing import *
from pathlib import Path
from configs import (
    EMBEDDING_MODEL,
    DEFAULT_VS_TYPE,
    KB_ROOT_PATH,
    VECTOR_SEARCH_TOP_K,
    logger,
    MODEL_LIST,
    KB_DICT,
    logging
)
import httpx
import requests
import asyncio
# from fastapi.responses import StreamingResponse
import json
from io import BytesIO

from streamlit_chatbox import *

def get_messages_history(chat_box: ChatBox, history_len: int) -> List[Dict]:
    def filter(msg):
        content = [x._content for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        return {
            "role": msg["role"],
            "content": content[0] if content else "",
        }

    history = chat_box.filter_history(100000, filter)  # workaround before upgrading streamlit-chatbox.
    user_count = 0
    i = 1
    for i in range(1, len(history) + 1):
        if history[-i]["role"] == "user":
            user_count += 1
            if user_count >= history_len:
                break
    return history[-i:]

def set_httpx_timeout(timeout=60.0):
    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

KB_ROOT_PATH = Path(KB_ROOT_PATH)
set_httpx_timeout()


class ApiRequest:
    '''
    1. Simplify API calling method
    2. Achieve no api call(TODO)
    '''
    def __init__(
        self,
        # base_url: str = "http://127.0.0.1:7861",
        llm_url: str,
        kb_url: str,
        timeout: float = 60.0,
        no_remote_api: bool = False,   # call api view function directly
    ):
        self.base_url = ""
        self.llm_url = llm_url
        self.kb_url = kb_url
        self.timeout = timeout
        self.no_remote_api = no_remote_api

    # def _parse_url(self, url: str) -> str:
    #     if (not url.startswith("http")
    #                 and self.base_url
    #             ):
    #         part1 = self.base_url.strip(" /")
    #         part2 = url.strip(" /")
    #         return f"{part1}/{part2}"
    #     else:
    #         return url
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

    # def _fastapi_stream2generator(self, response: StreamingResponse, as_json: bool =False):
    #     try:
    #         loop = asyncio.get_event_loop()
    #     except:
    #         loop = asyncio.new_event_loop()
        
    #     try:
    #         for chunk in  iter_over_async(response.body_iterator, loop):
    #             if as_json and chunk:
    #                 yield json.loads(chunk)
    #             elif chunk.strip():
    #                 yield chunk
    #     except Exception as e:
    #         logger.error(e)

    # model config request
    def reload_model_v1(
        self,
        model: str,
        model_dtype: str,
        framework: str
        ):
        pass

    
    def llm_chat_v1(
        self,
        model: str,
        prompt: str,
        top_p: float,
        temperature: float,
        max_token_length: int,
        history: List[Dict] = [],
        stream: bool = False
    ):
        data = {
            "prompt": prompt,
            "history": history,
            "top_p": top_p,
            "temperature": temperature,
            "max_token_length": max_token_length,
            "stream": stream,
        }
        url=self.llm_url+"/v1/completions"
        response = requests.post(
            url=url,
            headers = {"Content-Type": "application/json"},
            json=data
        )
        return response

    def knowledge_base_chat_v1(
        self,
        query: str,
        kb_name: str,
        kb_top_k: int = VECTOR_SEARCH_TOP_K,
        # score_threshold: float = SCORE_THRESHOLD,
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_token_length: int = 512,
        history: List[Dict] = [],
        stream: bool = True,
        no_remote_api: bool = False,
    ):
        if no_remote_api is None:
            no_remote_api = self.no_remote_api
        
        file=""
        if kb_name in KB_DICT:
            kb = KB_DICT[kb_name]
            if len(kb["files"]) > 0:
                file = kb["files"][0]
            else:
                raise Exception(f"{kb_name} contianes no file")
        else:
            raise Exception(f"{kb_name} not found")

        data = {
            "question": query,
            "file": file,
            # "score_threshold": score_threshold,
            "search_kwargs": {"k": kb_top_k},
            "temperature": temperature,
            "top_p": top_p,
            "max_token": max_token_length,
            "history": history,
            "with_history": "False",
            "stream": stream,
        }

        if no_remote_api:
            pass
        else:
            url=self.kb_url+"/query"
            response = requests.post(
                url=url,
                headers = {"Content-Type": "application/json"},
                json=data
            )
            return response.json()

    def knowledge_base_load_v1(
        self,
        question: str,
        selected_kb: str,
        max_token: int = 1000,
        top_k: int = VECTOR_SEARCH_TOP_K,
        top_p: int = 0.8,
        temperature: int = 0.5,
        # score_threshold: float = SCORE_THRESHOLD,
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
            # "score_threshold": score_threshold,
            "search_kwargs": {search_kwargs},
            "with_history": with_history,
            "history": history,
        }

        url=self.kb_url+"/load"
        response = requests.post(
            url=url,
            headers = {"Content-Type": "application/json"},
            json=data
        )
        return response
        
    def list_knowledge_bases_v1(
        self,
        no_remote_api: bool = None,
    ):
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        if no_remote_api:
            pass
        else:
            return KB_DICT
    
    # v1: just create kb in KB_DICT
    # TODO: wait for kb service support kn management 
    def create_knowledge_base_v1(
        self,
        kb_name: str,
        vs_type: str = "milvus",
        embed_model: str = EMBEDDING_MODEL,
        no_remote_api: bool = None,
    ):
        '''
        No support for kb management in kb server, just use KB_DICT store the kb info, real kb creat in upload_kb_doc_v1()
        '''
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        if no_remote_api:
            pass
        else:
            KB_DICT[kb_name]={
                "files": [],
                "vs_type": vs_type,
                "embed_model": embed_model,
            }
            response = {
                "msg": f"Create {kb_name} successfully"
            }
            return response
    # just delete kb from app RAM config and file path
    # TODO: delete file from kb
    def delete_knowledge_base_v1(
        self,
        kb_name: str,
        no_remote_api: bool = None,
    ):

        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        if no_remote_api:
            pass
        else:
            if kb_name in KB_DICT:
                for file in KB_DICT[kb_name]["files"]:
                    data = {
                        "file": file,
                        "embed_model": KB_DICT[kb_name]["embed_model"],
                    }
                    url=self.kb_url+"/delete"
                    ret = requests.post(
                        url=url,
                        headers = {"Content-Type": "application/json"},
                        json=data
                    )
                    # TODO: need update after kb server support one kb with multi files, delete file from server
                    file_path = os.path.join(KB_ROOT_PATH, file)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logging.info(f"{kb_name}: delete {file} successfully")
                        except Exception as e:
                            logging.error(f"{kb_name}: delete {file} error: {e}")
                    else:
                        logging.warning(f"{kb_name}: {file} not found")
                    # if ret.json()["status"] != 200:
                    #     response = {
                    #         "message": f"Delete {kb_name} error: {ret.json()['message']}",
                    #         "status": 200
                    #     }
                    #     return response
                del KB_DICT[kb_name]
                response = {
                    "message": f"Delete {kb_name} successfully",
                    "status": 200
                }
            else:
                response = {
                    "message": f"{kb_name} does't exist",
                    "status": 400
                }
            return response

    # v1: also means realy sent req to langchain to create a new knowledge base with only the 1st file
    # TODO: Upload file to existed kb in kb service if langchain app support multi files and support kb management
    def upload_kb_doc_v1(
        self,
        kb_name: str,
        file: Union[str, Path, bytes],
        vs_type: str = DEFAULT_VS_TYPE,
        embed_model: str = EMBEDDING_MODEL,
        no_remote_api: bool = None,
    ):
        if no_remote_api is None:
            no_remote_api = self.no_remote_api

        if isinstance(file, bytes): # raw bytes
            file = BytesIO(file)
        elif hasattr(file, "read"): # a file io like object
            filename = file.name
        else: # a local path
            file = Path(file).absolute().open("rb")
            filename = file.name
            
        if no_remote_api:
            pass
        else:
            # upload file to KB_ROOT_PATH
            filename = file.name
            file_path = os.path.join(KB_ROOT_PATH, filename)
            kb = KB_DICT[kb_name]
            if filename not in kb["files"]:
                with open(file_path, "wb") as f:
                    f.write(file.read())
                kb["files"].append(filename)

            # send request to kb server
            data = {
                # "kb_name": kb_name, # TODO: update ubtil server support kb with multi files
                "file": filename,
                # "vs_type": vs_type, # TODO: update ubtil server support multi vs_types
                "embed_model": embed_model,
            }
            url=self.kb_url+"/load"
            ret = requests.post(
                url=url,
                headers = {"Content-Type": "application/json"},
                json=data
            )
            if ret.json()["status"] == 201 or ret.json()["status"] == 200:
                logging.info(f"{kb_name}: upload {filename} to {kb_name} successfully")
                response = {
                    "message": f"Upload {filename} to {kb_name} successfully",
                    "status": 200
                }
            return response


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