import streamlit as st
from webui_pages.utils import *
from streamlit_chatbox import *
from datetime import datetime
from typing import List, Dict
import os

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "icon_blue_square.png"
    )
)


def get_messages_history(history_len: int) -> List[Dict]:
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


def dialogue_page(api: ApiRequest):
    chat_box.init_session()

    with st.sidebar:
        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"Changed to {mode} mode。"
            if mode == "knowledge base chat":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} current knowledge base: `{cur_kb}`。"
            st.toast(text)
            # sac.alert(text, description="descp", type="success", closable=True, banner=True)
        
        # def on_model_change():
        #     # todo: reload model if loaded model is diff
        #     st.toast(f"Load model {st.session_state.selected_model} successfully")

        dialogue_mode = st.selectbox("Select chat mode",
                                     ["LLM chat",
                                      "Knowledge base chat",
                                       "Search engine chat",
                                      ],
                                     on_change=on_mode_change,
                                     key="dialogue_mode",
                                     )
        history_len = st.number_input("Historical dialogue rounds:", 0, 10, 0)

        # todo: support history len

        def on_kb_change():
            st.toast(f"Load knowledge base： {st.session_state.selected_kb}")

        # with st.expander("LLM config", True):
        #     model_list = api.list_models(service="llm_service")
        #     selected_model = st.selectbox(
        #         "Please select model",
        #         model_list,
        #         on_change=on_model_change,
        #         key="selected_model",
        #     )
        #     # todo: selected_framework, dtype
        #     selected_model_dtype = st.selectbox(
        #         "Please select model",
        #         options=["FP16", "INT4"],
        #         # on_change=on_model_dtype_change,
        #         key="selected_model_dtype",
        #     )
        #     selected_framework = st.selectbox(
        #         "Please select model",
        #         options=["bigdl-llm", "transformers"],
        #         # on_change=on_framework_change,
        #         key="selected_framework",
        #     )
        kb_list=["1","2"]
        if dialogue_mode == "Knowledge base chat":
            with st.expander("Knowledge base config", True):
                # todo
                kb_list = api.list_knowledge_bases(no_remote_api=True)
                selected_kb = st.selectbox(
                    "Select Knowledge base",
                    kb_list,
                    on_change=on_kb_change,
                    key="selected_kb",
                )
                kb_top_k = st.number_input("Matched knowledge items: ", 1, 20, 2)
                score_threshold = st.number_input("Knowledge matching score threshold: ", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01)
                chunk_content = st.checkbox("use context chunk", False, disabled=True)
                chunk_size = st.slider("context chunk size：", 0, 500, 250, disabled=True)
        elif dialogue_mode == "Searah engine chat":
            with st.expander("Serach engine config", True):
                search_engine = st.selectbox("Select search engine", SEARCH_ENGINES.keys(), 0)
                se_top_k = st.number_input("Match results number", 1, 20, 3)

    # Display chat messages from history on app rerun

    chat_box.output_messages()

    chat_input_placeholder = "Please enter the dialogue content, use Ctrl+Enter for line break"

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        history = get_messages_history(history_len)
        chat_box.user_say(prompt)
        if dialogue_mode == "LLM chat":
            chat_box.ai_say("thinking...")
            text = ""
            # r = api.chat_chat(prompt, history)
            # for t in r:
            #     if error_msg := check_error_msg(t): # check whether error occured
            #         st.error(error_msg)
            #         break
            #     text += t
            #     chat_box.update_msg(text)
            
            # Todo: add temp, top_p, top_k, temperature, max_token_length
            ret = api.llm_chat(st.session_state.get("selected_model"), prompt, history)
            print(ret)
            if ret["status"] == 200:
                text = ret["completion"]
            else:
                text = "Error in llm server"
            chat_box.update_msg(text, streaming=False)  
        elif dialogue_mode == "knowledge base chat":
            history = get_messages_history(history_len)
            chat_box.ai_say([
                f"Querying knowledge base `{selected_kb}` ...",
                Markdown("...", in_expander=True, title="Knowledge base matching results"),
            ])
            text = ""
            ret = api.knowledge_base_chat_v1(
                question=prompt,
                selected_kb=selected_kb,
                kb_top_k=kb_top_k,
                # score_threshold=score_threshold,
                with_history=False)
            if ret["status"] == 200:
                text = ret["answer"]
            else:
                text = "Error in knowledge base chat server"
            chat_box.update_msg(text, 0, streaming=False)
        # elif dialogue_mode == "搜索引擎问答":
        #     chat_box.ai_say([
        #         f"正在执行 `{search_engine}` 搜索...",
        #         Markdown("...", in_expander=True, title="网络搜索结果"),
        #     ])
        #     text = ""
        #     for d in api.search_engine_chat(prompt, search_engine, se_top_k):
        #         if error_msg := check_error_msg(d): # check whether error occured
        #             st.error(error_msg)
        #         text += d["answer"]
        #         chat_box.update_msg(text, 0)
        #         chat_box.update_msg("\n\n".join(d["docs"]), 1, streaming=False)
        #     chat_box.update_msg(text, 0, streaming=False)

    now = datetime.now()
    with st.sidebar:
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)

    with st.sidebar:

        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "Clear",
                use_container_width=True,
        ):
            chat_box.reset_history()
            st.experimental_rerun()

    export_btn.download_button(
        "Export",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_chat.md",
        mime="text/markdown",
        use_container_width=True,
    )
