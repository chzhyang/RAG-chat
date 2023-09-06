# Usage：
# 1. pip install streamlit-option-menu streamlit-chatbox>=1.1.6
# 2. startup langchain-app or llm server
# 3. streamlit run webui.py --server.port 8000

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)

import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages import *

from configs import VERSION, LLM_SERVICE_URL, KB_SERVICE_URL, LLM_MODEL

api = ApiRequest(llm_url=LLM_SERVICE_URL, kb_url=KB_SERVICE_URL, no_remote_api=False)

if __name__ == "__main__":
    st.set_page_config(
        "LLM-Chatbot",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/chzhyang',
            'About': f""" Welcome to LLM-Chatbot {VERSION}"""
        }
    )

    if not chat_box.chat_inited:
        st.toast(
            f"Welcome to [LLM-Chatbot](https://github.com/chzhyang/) \n"
            f"Current LLM is `{LLM_MODEL}`"
        )

    pages = {
        "Chat": {
            "icon": "chat",
            "func": dialogue_page,
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

    if selected_page in pages:
        pages[selected_page]["func"](api)
