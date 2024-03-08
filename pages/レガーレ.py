import os
import streamlit as st
import pinecone
import logging
import sys
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain import OpenAI as OpenAIllm
from llama_index.vector_stores import PineconeVectorStore
from llama_index import StorageContext, ServiceContext, GPTVectorStoreIndex, load_index_from_storage, QuestionAnswerPrompt
from llama_index.llms import OpenAI
from llama_index.callbacks import CallbackManager, LlamaDebugHandler


load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    

def make_engin():
    pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    pinecone_index = pinecone.Index(os.environ["PINECONE_NAME"])
    vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    )
    qa_template = QuestionAnswerPrompt("""<s>[INST] <<SYS>>
    あなたはヘルプセンターのオペレターです。質問に対して親切に答えください。
    回答は全て日本語で答えてください。
    <</SYS>>
    == 以下にコンテキスト情報を提供します。
    {context_str}
    == 質問
    {query_str}
    [/INST]
    """)
    llm = OpenAI(model=os.environ["OPENAI_API_MODEL"], temperature=float(os.environ["OPENAI_API_TEMPERATURE"]), max_tokens=int(os.environ["OPENAI_API_MAX_TOKENS"]))
    llama_debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([llama_debug_handler])
    service_context = ServiceContext.from_defaults(llm=llm, callback_manager=callback_manager)
    index = GPTVectorStoreIndex.from_vector_store(
      vector_store,
      service_context=service_context,
      embed_model=OpenAIEmbeddings(model=os.environ["OPENAI_API_EMBEDDING"])
    )
    query_engine = index.as_query_engine(
    similarity_top_k=int(os.environ["SIMILARITY_TOP_K"]),
    text_qa_template=qa_template,
    streaming=True,
    )
    return query_engine


if "query_engine" not in st.session_state:
    st.session_state.query_engine = make_engin()

st.title("Legare")

st.sidebar.markdown("[使い方](https://github.com/dice-tfukuda/dice-demo/issues/4)")
if st.sidebar.button('gpt-4'):
  os.environ["OPENAI_API_MODEL"] = "gpt-4"
  st.session_state.query_engine = make_engin()
if st.sidebar.button('gpt-3.5-turbo'):
  os.environ["OPENAI_API_MODEL"] = "gpt-3.5-turbo"
  st.session_state.query_engine = make_engin()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result_area = st.empty()
        # info_area = st.empty()
        response = st.session_state.query_engine.query(prompt)
        text = ''
        for next_text in response.response_gen:
            text += next_text
            result_area.write(text)

    st.session_state.messages.append({"role": "assistant", "content": text})