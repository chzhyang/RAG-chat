# Usage：
# 1. pip install streamlit-option-menu streamlit-chatbox>=1.1.6
# 2. startup langchain-app and llm server
# 3. streamlit run webui.py --server.port 7860

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)

import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages import *

from configs import VERSION, LLM_SERVICE_URL, KB_SERVICE_URL, MODEL_CONFIG

api = ApiRequest(llm_url=LLM_SERVICE_URL, kb_url=KB_SERVICE_URL, no_remote_api=False)

session_state = st.session_state
if "selected_model" not in session_state:
    session_state.selected_model = MODEL_CONFIG["model"]

if __name__ == "__main__":
    st.set_page_config(
        "Cloud-LLM-Chatbot",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/chatchat-space/Langchain-Chatchat',
            # 'Report a bug': "https://github.com/chatchat-space/Langchain-Chatchat/issues",
            'About': f""" Welcome to Cloud-LLM-Chatbot {VERSION}"""
        }
    )

    if not chat_box.chat_inited:
        st.toast(
            f"Welcome to [Cloud-LLM-Chatbot](https://github.com/chzhyang/) \n\n"
            f'Current model: `{MODEL_CONFIG["model"]}`'
        )

    pages = {
        "LLM Chat": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "Knowledge Base Chat": {
            "icon": "chat",
            "func": knowledge_base_page,
        },
        "Model Config": {
            "icon": "hdd-stack",
            "func": model_config_page,
        },
        "Knowladge Base Config": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },
    }

    with st.sidebar:
        st.title('LLM Chatbot')
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )

        MODEL_CONFIG["temperature"] = st.slider(
            'temperature',
            min_value=0.01,
            max_value=5.0,
            value=0.9,
            step=0.1,
        )
        MODEL_CONFIG["top_p"] = st.slider(
            'top_p',
            min_value=0.01,
            max_value=1.0,
            value=0.7,
            step=0.1,
        )
        MODEL_CONFIG["max_length"] = st.slider(
            'max_length',
            min_value=32,
            max_value=2048,
            value=512,
            step=8,
        )

    if selected_page in pages:
        pages[selected_page]["func"](api)