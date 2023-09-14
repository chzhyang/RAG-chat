import streamlit as st
from webui_pages.utils import *
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
from server.knowledge_base.utils import get_file_path, LOADER_DICT
from server.knowledge_base.kb_service.base import get_kb_details, get_kb_file_details
from typing import Literal, Dict, Tuple
from configs.model_config import embedding_model_dict, KB_LIST, KB_VS_CONFIG, EMBEDDING_MODEL, DEFAULT_VS_TYPE, LOADER_DICT_V1
import os
import time


# SENTENCE_SIZE = 100

cell_renderer = JsCode("""function(params) {if(params.value==true){return '✓'}else{return '×'}}""")


def config_aggrid(
        df: pd.DataFrame,
        columns: Dict[Tuple[str, str], Dict] = {},
        selection_mode: Literal["single", "multiple", "disabled"] = "single",
        use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        # pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    return gb


def file_exists(kb: str, selected_rows: List) -> Tuple[str, str]:
    '''
    check whether a doc file exists in local knowledge base folder.
    return the file's name and path if it exists.
    '''
    if selected_rows:
        file_name = selected_rows[0]["file_name"]
        file_path = get_file_path(kb, file_name)
        if os.path.isfile(file_path):
            return file_name, file_path
    return "", ""


def knowledge_base_config_page(api: ApiRequest):
    try:
        # kb_list = {x["kb_name"]: x for x in get_kb_details()}
        kb_list = {x["kb_name"]: x for x in KB_LIST}
    except Exception as e:
        st.error("Failed to startup knowledge base")
        st.stop()
    kb_names = list(kb_list.keys())

    if "selected_kb_name" in st.session_state and st.session_state["selected_kb_name"] in kb_names:
        selected_kb_index = kb_names.index(st.session_state["selected_kb_name"])
    else:
        selected_kb_index = 0

    def format_selected_kb(kb_name: str) -> str:
        if kb := kb_list.get(kb_name):
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
                placeholder="Just support English",
                key="kb_name",
            )

            cols = st.columns(2)

            vs_types = list(KB_VS_CONFIG.keys())
            vs_type = cols[0].selectbox(
                "Vector store type",
                vs_types,
                index=vs_types.index(DEFAULT_VS_TYPE),
                key="vs_type",
            )

            embed_models = list(embedding_model_dict.keys())

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
            elif kb_name in kb_list:
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
        kb = selected_kb
        # upload doc to target knowledge base
        # sentence_size = st.slider("sentence size limit to fit vector store", 1, 1000, SENTENCE_SIZE, disabled=True)
        files = st.file_uploader(
            "upload files",
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
            data = [{"file": f, "kb_name": kb, "not_refresh_vs_cache": True} for f in files]
            data[-1]["not_refresh_vs_cache"]=False
            # v1: just support only 1 file
            for k in data:
                ret = api.upload_kb_doc_v1(**k)
                if msg := check_success_msg(ret):
                    st.toast(msg, icon="✔")
                elif msg := check_error_msg(ret):
                    st.toast(msg, icon="✖")
            st.session_state.files = []

        if cols[2].button(
                "Delete knowledge base",
                use_container_width=True,
        ):
            ret = api.delete_knowledge_base_v1(kb)
            st.toast(ret.get("msg", " "))
            time.sleep(1)
            st.experimental_rerun()
        # st.divider()

        # # Knowledge base details
        # doc_details = pd.DataFrame(get_kb_file_details(kb))
        # if not len(doc_details):
        #     st.info(f"知识库 `{kb}` 中暂无文件")
        # else:
        #     st.write(f"知识库 `{kb}` 中已有文件:")
        #     st.info("知识库中包含源文件与向量库，请从下表中选择文件后操作")
        #     doc_details.drop(columns=["kb_name"], inplace=True)
        #     doc_details = doc_details[[
        #         "No", "file_name", "document_loader", "docs_count", "in_folder", "in_db",
        #     ]]
        #     # doc_details["in_folder"] = doc_details["in_folder"].replace(True, "✓").replace(False, "×")
        #     # doc_details["in_db"] = doc_details["in_db"].replace(True, "✓").replace(False, "×")
        #     gb = config_aggrid(
        #         doc_details,
        #         {
        #             ("No", "序号"): {},
        #             ("file_name", "文档名称"): {},
        #             # ("file_ext", "文档类型"): {},
        #             # ("file_version", "文档版本"): {},
        #             ("document_loader", "文档加载器"): {},
        #             ("docs_count", "文档数量"): {},
        #             # ("text_splitter", "分词器"): {},
        #             # ("create_time", "创建时间"): {},
        #             ("in_folder", "源文件"): {"cellRenderer": cell_renderer},
        #             ("in_db", "向量库"): {"cellRenderer": cell_renderer},
        #         },
        #         "multiple",
        #     )

        #     doc_grid = AgGrid(
        #         doc_details,
        #         gb.build(),
        #         columns_auto_size_mode="FIT_CONTENTS",
        #         theme="alpine",
        #         custom_css={
        #             "#gridToolBar": {"display": "none"},
        #         },
        #         allow_unsafe_jscode=True,
        #         enable_enterprise_modules=False
        #     )

        #     selected_rows = doc_grid.get("selected_rows", [])

        #     cols = st.columns(4)
        #     file_name, file_path = file_exists(kb, selected_rows)
        #     if file_path:
        #         with open(file_path, "rb") as fp:
        #             cols[0].download_button(
        #                 "下载选中文档",
        #                 fp,
        #                 file_name=file_name,
        #                 use_container_width=True, )
        #     else:
        #         cols[0].download_button(
        #             "下载选中文档",
        #             "",
        #             disabled=True,
        #             use_container_width=True, )

        #     st.write()
        #     # 将文件分词并加载到向量库中
        #     if cols[1].button(
        #             "重新添加至向量库" if selected_rows and (pd.DataFrame(selected_rows)["in_db"]).any() else "添加至向量库",
        #             disabled=not file_exists(kb, selected_rows)[0],
        #             use_container_width=True,
        #     ):
        #         for row in selected_rows:
        #             api.update_kb_doc(kb, row["file_name"])
        #         st.experimental_rerun()

        #     # 将文件从向量库中删除，但不删除文件本身。
        #     if cols[2].button(
        #             "从向量库删除",
        #             disabled=not (selected_rows and selected_rows[0]["in_db"]),
        #             use_container_width=True,
        #     ):
        #         for row in selected_rows:
        #             api.delete_kb_doc(kb, row["file_name"])
        #         st.experimental_rerun()

        #     if cols[3].button(
        #             "从知识库中删除",
        #             type="primary",
        #             use_container_width=True,
        #     ):
        #         for row in selected_rows:
        #             ret = api.delete_kb_doc(kb, row["file_name"], True)
        #             st.toast(ret.get("msg", " "))
        #         st.experimental_rerun()

        # st.divider()

        # cols = st.columns(3)

        # if cols[0].button(
        #         "依据源文件重建向量库",
        #         # help="无需上传文件，通过其它方式将文档拷贝到对应知识库content目录下，点击本按钮即可重建知识库。",
        #         use_container_width=True,
        #         type="primary",
        # ):
        #     with st.spinner("向量库重构中，请耐心等待，勿刷新或关闭页面。"):
        #         empty = st.empty()
        #         empty.progress(0.0, "")
        #         for d in api.recreate_vector_store(kb):
        #             if msg := check_error_msg(d):
        #                 st.toast(msg)
        #             else:
        #                 empty.progress(d["finished"] / d["total"], d["msg"])
        #         st.experimental_rerun()

        # if cols[2].button(
        #         "删除知识库",
        #         use_container_width=True,
        # ):
        #     ret = api.delete_knowledge_base(kb)
        #     st.toast(ret.get("msg", " "))
        #     time.sleep(1)
        #     st.experimental_rerun()
