import streamlit as st
from webui_pages.utils import *
from streamlit_chatbox import *
from datetime import datetime
from server.chat.search_engine_chat import SEARCH_ENGINES
from typing import List, Dict
import os

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "../img",
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
    from configs import MODEL_CONFIG
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

        dialogue_mode = st.selectbox("Select chat mode",
                                     ["LLM chat",
                                      "Knowledge base chat",
                                    #   "搜索引擎问答",
                                      ],
                                     on_change=on_mode_change,
                                     key="dialogue_mode",
                                     )
        history_len = st.number_input("Historical dialogue rounds:", 0, 10, 0)

        # todo: support history len

        def on_kb_change():
            st.toast(f"已加载知识库： {st.session_state.selected_kb}")

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
                # score_threshold = st.number_input("Knowledge matching score threshold: ", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01)
                # chunk_content = st.checkbox("关联上下文", False, disabled=True)
                # chunk_size = st.slider("关联长度：", 0, 500, 250, disabled=True)
        # elif dialogue_mode == "搜索引擎问答":
        #     with st.expander("搜索引擎配置", True):
        #         search_engine = st.selectbox("请选择搜索引擎", SEARCH_ENGINES.keys(), 0)
        #         se_top_k = st.number_input("匹配搜索结果条数：", 1, 20, 3)

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
            
            ret = api.llm_chat_v1(
                    # model=str(config["model"]),
                    prompt=prompt,
                    history=history,
                    top_p=float(MODEL_CONFIG["top_p"]),
                    temperature=float(MODEL_CONFIG["temperature"]),
                    max_token_length=int(MODEL_CONFIG["max_length"])
                )
            if ret.status_code == 200:
                parsed_ret = ret.json()
                text = parsed_ret["completion"]
            else:
                text = "Error in llm server"
            chat_box.update_msg(element=text, streaming=False)  # 更新最终的字符串，去除光标
        elif dialogue_mode == "knowledge base chat":
            history = get_messages_history(history_len)
            chat_box.ai_say([
                f"Querying knowledge base `{selected_kb}` ...",
                Markdown("...", in_expander=True, title="Knowledge base matching results"),
            ])
            text = ""
            # for d in api.knowledge_base_chat(prompt, selected_kb, kb_top_k, score_threshold, history):
            #     if error_msg := check_error_msg(d): # check whether error occured
            #         st.error(error_msg)
            #     text += d["answer"]
            #     chat_box.update_msg(text, 0)
            #     chat_box.update_msg("\n\n".join(d["docs"]), 1, streaming=False)
            # chat_box.update_msg(text, 0, streaming=False)
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

    now = datetime.now()

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
