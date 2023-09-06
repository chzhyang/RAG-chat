import streamlit as st
from web.webui_pages.utils import *

def model_config_page(api: ApiRequest):
    # model_list = api.list_models(service="llm_service")
    model_list = ["chatglm2-6b", "llama2-7b"]
    datatype_list = ["fp16", "int4"]
    framework_list = ["openvino", "transformers"]

    selected_model = st.selectbox(
                "Please select model",
                model_list,
                key="selected_model",
            )
    selected_datatype  = st.selectbox(
                "Please select model datatype",
                datatype_list,
                key="selected_datatype",
            )
    selected_framework = st.selectbox(
                "Please select framework",
                framework_list,
                key="selected_framework",
            )
    
    if st.button("Submit model config"):
        if selected_model != st.session_state.selected_model or selected_datatype != st.session_state.selected_datatype or selected_framework != st.session_state.selected_framework:
            ret = api.reload_model(model=selected_model, model_dtype=selected_datatype, framework=selected_framework)
            if ret["status"] == "200":
                st.session_state.selected_model = selected_model
                st.session_state.selected_datatype = selected_datatype
                st.session_state.selected_framework = selected_framework
                st.toast(f"Load model {st.session_state.selected_model} - {st.session_state.selected_datatype} successfully, and will run inference on {st.session_state.selected_framework}")
            else:
                st.toast(f"Error in reload model")
        else:
            st.toast(f"Current model config is already loaded")
