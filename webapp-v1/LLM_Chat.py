# Usage：
# 1. pip install streamlit-option-menu streamlit-chatbox>=1.1.6
# 2. startup langchain-app and llm server
# 3. streamlit run LLM_Chat.py --server.port 7860

from datetime import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)

import streamlit as st
from utils import *
# from streamlit_option_menu import option_menu
from pages import *
from streamlit_chatbox import *
from configs import (
    LLM_SERVICE_URL, 
    KB_SERVICE_URL, 
    MODEL_CONFIG, 
    MODEL_LIST, 
    MODEL_DTYPE_LIST, 
    FRAMEWORK_LIST
)

# st.set_page_config(
#     "Cloud-LLM-Chat",
#     # os.path.join("img", "chatchat_icon_blue_square_v2.png"),
#     initial_sidebar_state="expanded",
#     # menu_items={
#     #     'Get Help': 'https://github.com/chatchat-space/Langchain-Chatchat',
#     #     # 'Report a bug': "https://github.com/chatchat-space/Langchain-Chatchat/issues",
#     #     'About': f""" Welcome to Cloud-LLM-Chatbot {VERSION}"""
#     # }
# )

api = ApiRequest(llm_url=LLM_SERVICE_URL, kb_url=KB_SERVICE_URL, no_remote_api=False)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_CONFIG["model"]

# st.title('Cloud LLM Chat')
if 'assistant_avatar_path' not in st.session_state:
    st.session_state.assistant_avatar_path = os.path.abspath(os.path.join(
        "./img",
        "icon_blue_square.png"
    ))
if "chat_box" not in st.session_state:
    st.session_state.chat_box = ChatBox(
        assistant_avatar=st.session_state.assistant_avatar_path
    )
    st.session_state.chat_box.init_session()
    st.session_state.chat_box.history.clear()
# logging.info(f"before init Inference config:\n {st.session_state.temperature}\n {st.session_state.top_p}\n {st.session_state.max_length}")
if 'temperature' not in st.session_state:
    st.session_state.temperature = MODEL_CONFIG["temperature"]

if 'top_p' not in st.session_state:
    st.session_state.top_p = MODEL_CONFIG["top_p"]

if 'max_length' not in st.session_state:
    st.session_state.max_length = MODEL_CONFIG["max_length"]

if 'history_len' not in st.session_state:
    st.session_state.history_len = MODEL_CONFIG["history_len"]

with st.sidebar:
    with st.expander("Inference config", False):
        temperature = st.slider(
            'temperature',
            min_value=0.01,
            max_value=5.0,
            value=st.session_state.get("temperature", MODEL_CONFIG["temperature"]),
            step=0.1,
            key="llm_temperature"
        )
        top_p = st.slider(
            'top_p',
            min_value=0.01,
            max_value=1.0,
            value=st.session_state.get("top_p", MODEL_CONFIG["top_p"]),
            step=0.1,
            key="llm_top_p"
        )
        max_length = st.slider(
            'max_length',
            min_value=32,
            max_value=2048,
            value=st.session_state.get("max_length", MODEL_CONFIG["max_length"]),
            step=8,
            key="llm_max_length"
        )

        history_len = st.number_input(
            "Historical dialogue rounds:", 
            min_value=0, 
            max_value=10, 
            value=0,
            key="llm_history_len"
        )
    
        st.session_state.history_len = history_len
        st.session_state.temperature = temperature
        st.session_state.top_p = top_p
        st.session_state.max_length = max_length
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
    cols = st.columns(2)
    export_btn = cols[0]
    if cols[1].button(
            "Clear",
            use_container_width=True,
            key="clear"
    ):
        st.session_state.chat_box.reset_history()
        st.experimental_rerun()

now = datetime.now()
export_btn.download_button(
        "Export",
        "".join(st.session_state.chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_chat.md",
        mime="text/markdown",
        use_container_width=True,
        key="download"
    )

st.session_state.chat_box.output_messages()

chat_input_placeholder = "Please enter the dialogue content, use Ctrl+Enter for line break"

if prompt := st.chat_input(chat_input_placeholder, key="llm_chat_prompt"):
    history = get_messages_history(st.session_state.chat_box, history_len)
    st.session_state.chat_box.user_say(prompt)
    st.session_state.chat_box.ai_say("thinking...")
    text = ""
    ret = api.llm_chat_v1(
            model=str(MODEL_CONFIG["model"]),
            prompt=prompt,
            history=history,
            top_p=float(top_p),
            temperature=float(temperature),
            max_token_length=int(max_length)
        )
    ret = ret.json()
    if ret["status"] == 200:
        text = ret["completion"]
    else:
        text = "Error in llm server"
    st.session_state.chat_box.update_msg(element=text, streaming=False)