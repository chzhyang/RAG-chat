# Web App for LLM Chat and Knowledge Base chat on Cloud

- LLM chat page(home): LLM chat with model config, inference config
- Knowledge base chat page: Knowledge base chat with knowledge base config, model config, inference config

![LLM-chat-screen](img/screen-llm-chat_2023-09-14.png)

![Knowledge-base-chat-screen](img/screen-kb-chat_2023-09-14.png)

![Knowledge-base-config-screen](img/screen-kb-config_2023-09-14.png)

## Usage

### Deploy locally

- Startup LLM server and knowledge base(langchain) server
- Config server url, MODEL_CONFIG, KB_DICT, KB_ROOT_PATH
- Run webapp: `streamlit run LLM_Chat.py --server.port 7860`

### Deploy with docker

- Build local image or `docker pull chzhyang/cloud-llm-webapp:v1`

    ```shell
    docker build --build-arg HTTP_PROXY==http://proxy-dmz.intel.com:911 \
    --build-arg HTTPS_PROXY=http://proxy-dmz.intel.com:912 \
    --build-arg NO_PROXY=localhost,127.0.0.1 \
    -f Dockerfile.webapp \
    -t chzhyang/cloud-llm-webapp:v1 .
    ```

- Startup container

    ```shell
    $ docker run -p 7860:7860 chzhyang/cloud-llm-webapp:v1
     
        You can now view your Streamlit app in your browser.
    ```

### Deploy with docker compose

- Build local image or `docker pull chzhyang/cloud-llm-webapp:v1`
- Update `docker-compose-webapp-allinone.yaml`
- Run docker-compose: `docker-compose -f docker-compose-webapp-allinone.yaml up`

### Deploy on K8s

