import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

st.set_page_config(page_title="Training AI Assistant")

st.title("AI Training Assistant")

@st.cache_resource
def load_engine():

    documents = SimpleDirectoryReader("qa_blocks").load_data()

    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.llm = OpenAI(model="gpt-5")

    index = VectorStoreIndex(nodes)

    return index.as_query_engine(similarity_top_k=5)

query_engine = load_engine()

question = st.text_input("Ask a business or marketing question:")

if question:
    response = query_engine.query(question)
    st.write(response)
