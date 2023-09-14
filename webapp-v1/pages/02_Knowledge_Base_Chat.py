import time
import streamlit as st
from utils import *
from datetime import datetime
from typing import Literal, Dict, Tuple
from configs import (
    EMBEDDING_MODEL_DICT, 
    LLM_SERVICE_URL, 
    KB_SERVICE_URL, 
    KB_VS_DICT, 
    EMBEDDING_MODEL, 
    DEFAULT_VS_TYPE, 
    LOADER_DICT_V1,
    MODEL_CONFIG, 
    MODEL_LIST, 
    MODEL_DTYPE_LIST, 
    FRAMEWORK_LIST
)
from streamlit_chatbox import *

# st.title('Cloud Knowledge Base Chat')
api = ApiRequest(llm_url=LLM_SERVICE_URL, kb_url=KB_SERVICE_URL, no_remote_api=False)

if "kb_chat_box" not in st.session_state:
    st.session_state.kb_chat_box = ChatBox(
        assistant_avatar=st.session_state.assistant_avatar_path
    )
    st.session_state.kb_chat_box.init_session()
    st.session_state.kb_chat_box.history.clear()

# if not kb_chat_box.chat_inited:
#     st.toast(
#         f"Welcome to [Cloud-LLM-Chatbot](https://github.com/chzhyang/) \n\n"
#         f'Current model: `{MODEL_CONFIG["model"]}`'
#     )

current_page = st.session_state.get("current_page", "chat")
show_chatbox = st.session_state.get("show_chatbox", True)

with st.sidebar:
    def on_kb_change():
        st.toast(f"Load knowledge base: `{st.session_state.selected_kb}`")

    kb_dict = api.list_knowledge_bases_v1(no_remote_api=False)
    kb_names = list(kb_dict.keys())
    selected_kb = st.selectbox(
        "Select Knowledge Base",
        kb_names,
        on_change=on_kb_change,
        key="selected_kb",
    )
    kb_top_k = st.number_input("Matched Knowledge Items: ", 1, 10, 2)
    
    # switch current page [chat, config]
    if st.button(
        "Knowledge Base Config",
        use_container_width=True,
    ):
        if current_page == "chat":
            current_page = "config"
            show_chatbox = False
        else:
            current_page = "chat"
            show_chatbox = True

    with st.expander("Inference Config", False):
        temperature = st.slider(
            'temperature',
            min_value=0.01,
            max_value=5.0,
            value=st.session_state.temperature,
            step=0.1,
            key="kb_temperature"
        )
        top_p = st.slider(
            'top_p',
            min_value=0.01,
            max_value=1.0,
            value=st.session_state.top_p,
            step=0.1,
            key="kb_top_p"
        )
        max_length = st.slider(
            'max_length',
            min_value=32,
            max_value=2048,
            value=st.session_state.max_length,
            step=8,
            key="kb_max_length"
        )

        history_len = st.number_input(
            "Historical dialogue rounds:",
            min_value=0, 
            max_value=10, 
            value=st.session_state.history_len,
            key="kb_history_len_key"
        )
    with st.expander("Model config", False):
        selected_model  = st.selectbox(
                "Please select model",
                MODEL_LIST,
                index=MODEL_LIST.index(MODEL_CONFIG["model"]),
            )
        selected_datatype  = st.selectbox(
                    "Please select model data type",
                    MODEL_DTYPE_LIST,
                    index=MODEL_DTYPE_LIST.index(MODEL_CONFIG["datatype"]),
                )
        selected_framework = st.selectbox(
                    "Please select framework",
                    FRAMEWORK_LIST,
                    index=FRAMEWORK_LIST.index(MODEL_CONFIG["framework"]),
                )
        
        if st.button("Submit model config", use_container_width=True):
            if selected_model != MODEL_CONFIG["model"] or selected_datatype != MODEL_CONFIG["datatype"] or selected_framework != MODEL_CONFIG["framework"]:
                MODEL_CONFIG["model"] = selected_model
                MODEL_CONFIG["datatype"] = selected_datatype
                MODEL_CONFIG["framework"] = selected_framework
                ret = api.reload_model_v1(model=selected_model, model_dtype=selected_datatype, framework=selected_framework)
                # if ret["status"] == "200":
                #     config["model"] = selected_model
                #     config["dtype"] = selected_datatype
                #     config["framework"] = selected_framework
                #     st.toast(f"""Load model {selected_model} successfully\n
                #             current dtype: {selected_datatype}\n
                #             current backend: {selected_framework}""")
                # else:
                #     st.toast(f"Error in reloading model")
            # else:
            #     st.toast(f"Current model config is already loaded")
    st.session_state.history_len = history_len
    st.session_state.temperature = temperature
    st.session_state.top_p = top_p
    st.session_state.max_length = max_length

    cols = st.columns(2)
    export_btn = cols[0]
    if cols[1].button(
            "Clear",
            use_container_width=True,
            key="kb_clear"
    ):
        st.session_state.kb_chat_box.reset_history()
        st.experimental_rerun()
now = datetime.now()
export_btn.download_button(
        "Export",
        "".join(st.session_state.kb_chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_chat.md",
        mime="text/markdown",
        use_container_width=True,
        key="kb_download"
    )

if current_page == "chat" and show_chatbox:
    st.session_state.kb_chat_box.output_messages()
    chat_input_placeholder = "Please enter the dialogue content, use Ctrl+Enter for line break"

    if prompt := st.chat_input(chat_input_placeholder, key="kb_chat_prompt"):
        history = get_messages_history(st.session_state.kb_chat_box, history_len)
        st.session_state.kb_chat_box.ai_say([
            f"Querying knowledge base `{selected_kb}` ...",
            Markdown("...", in_expander=True, title="Knowledge base matching results"),
        ])
        text = ""
        ret = api.knowledge_base_chat_v1(
            query=prompt,
            kb_name=selected_kb,
            kb_top_k=kb_top_k,
            # score_threshold=score_threshold,
            temperature=temperature,
            top_p=top_p,
            max_token_length=max_length,
            history=history,
            stream=False
        )
        if ret["status"] == 200:
            text = ret["answer"]
        else:
            text = f'Error in knowledge base chat server: {ret["message"]}'
        st.session_state.kb_chat_box.update_msg(text, 0, streaming=False)

if current_page == "config":
    try:
        # kb_list = {x["kb_name"]: x for x in get_kb_details()}
        kb_dict = api.list_knowledge_bases_v1()
    except Exception as e:
        st.error("Failed to startup knowledge base")
        st.stop()
    kb_names = list(kb_dict.keys())

    if "selected_kb_name" in st.session_state and st.session_state["selected_kb_name"] in kb_names:
        selected_kb_index = kb_names.index(st.session_state["selected_kb_name"])
    else:
        selected_kb_index = 0

    def format_selected_kb(kb_name: str) -> str:
        if kb := kb_dict.get(kb_name):
            return f"{kb_name} ({kb['vs_type']} @ {kb['embed_model']})"
        else:
            return kb_name

    selected_kb = st.selectbox(
        "Please select or create a knowledge base:",
        kb_names + ["create knowledge base"],
        format_func=format_selected_kb,
        index=selected_kb_index
    )

    if selected_kb == "create knowledge base":
        with st.form("create knowledge base"):

            kb_name = st.text_input(
                "Knowledge base name",
                placeholder="Support English only",
                key="kb_name",
            )

            cols = st.columns(2)

            vs_types = list(KB_VS_DICT.keys())
            vs_type = cols[0].selectbox(
                "Vector store type",
                vs_types,
                index=vs_types.index(DEFAULT_VS_TYPE),
                key="vs_type",
            )

            embed_models = list(EMBEDDING_MODEL_DICT.keys())

            embed_model = cols[1].selectbox(
                "Embedding model",
                embed_models,
                index=embed_models.index(EMBEDDING_MODEL),
                key="embed_model",
            )

            submit_create_kb = st.form_submit_button(
                "Create",
                # disabled=not bool(kb_name),
                use_container_width=True,
            )
        # create knowledge base
        if submit_create_kb:
            if not kb_name or not kb_name.strip():
                st.error(f"knowledge name is None!")
            elif kb_name in kb_dict:
                st.error(f"{kb_name} already exists!")
            else:
                ret = api.create_knowledge_base_v1(
                    kb_name=kb_name,
                    vs_type=vs_type,
                    embed_model=embed_model,
                )
                st.toast(ret["msg"])
                st.session_state["selected_kb_name"] = kb_name
                st.experimental_rerun()
    elif selected_kb:
        kb_name = selected_kb
        st.write(f"`{kb_name}` details:")
        doc_list = KB_DICT[kb_name]["files"]
        items = ""
        for i, doc in enumerate(doc_list):
            if i == 0:
                items += f"{doc}"
            else:
                items += f", {doc}"
        # add /t(tab) in f""

        st.info(f'''
            Embedding Model:   {KB_DICT[kb_name]["embed_model"]}\n
            Vector Store:      {KB_DICT[kb_name]["vs_type"]}\n
            File Count:        {len(doc_list)}\n
            File List:         {items}
        ''')
        # upload doc to target knowledge base
        # sentence_size = st.slider("sentence size limit to fit vector store", 1, 1000, SENTENCE_SIZE, disabled=True)
        files = st.file_uploader(
            "Upload files",
            [i for ls in LOADER_DICT_V1.values() for i in ls],
            accept_multiple_files=True,
        )
        cols = st.columns(3)

        if cols[0].button(
                "Add files to knowledge base",
                help="upload before add",
                # use_container_width=True,
                disabled=len(files) == 0,
        ):
            # v1: just support only 1 file
            for file in files:
                ret = api.upload_kb_doc_v1(kb_name, file)
                if ret["status"] == 200:
                    st.toast(ret["message"], icon="✔")
                else:
                    st.toast(ret["message"], icon="✖")
            time.sleep(2)
            st.experimental_rerun()
            # st.session_state.files = []

        if cols[1].button(
                "Delete knowledge base",
                use_container_width=True,
        ):
            ret = api.delete_knowledge_base_v1(kb_name)
            if ret["status"] == 200:
                st.toast(ret["message"], icon="✔")
            else:
                st.toast(ret["message"], icon="✖")
            time.sleep(2)
            st.experimental_rerun()
        # switch current page to chat
        if cols[2].button(
            "Exit",
            use_container_width=True,
        ):
            if current_page == "config":
                current_page = "chat"
                show_chatbox = True

st.session_state.current_page = current_page