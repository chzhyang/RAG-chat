import streamlit as st
from web.webui_pages.utils import *

def model_config_page(api: ApiRequest):
    from configs import MODEL_CONFIG, MODEL_LIST, MODEL_DTYPE_LIST, FRAMEWORK_LIST
    
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
    
    if st.button("Submit model config"):
        if selected_model != MODEL_CONFIG["model"] or selected_datatype != MODEL_CONFIG["datatype"] or selected_framework != MODEL_CONFIG["framework"]:
            MODEL_CONFIG["model"] = selected_model
            MODEL_CONFIG["datatype"] = selected_datatype
            MODEL_CONFIG["framework"] = selected_framework
            ret = api.reload_model(model=selected_model, model_dtype=selected_datatype, framework=selected_framework)
            # if ret["status"] == "200":
            #     config["model"] = selected_model
            #     config["dtype"] = selected_datatype
            #     config["framework"] = selected_framework
            #     st.toast(f"""Load model {selected_model} successfully\n
            #             current dtype: {selected_datatype}\n
            #             current backend: {selected_framework}""")
            # else:
            #     st.toast(f"Error in reloading model")
        else:
            st.toast(f"Current model config is already loaded")