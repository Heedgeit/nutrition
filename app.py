import streamlit as st
import os
import time
import dotenv

from chroma import ChromaVectorStore
from reasoning import GroqReasoningModel
from groq import Groq

dotenv.load_dotenv()


# -----------------------
# Cached System Loader
# -----------------------

@st.cache_resource
def load_system():
    vector_store = ChromaVectorStore(db_path="./chroma_db")
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    llm = GroqReasoningModel(groq)
    return vector_store, llm


vector_store, llm = load_system()


# -----------------------
# RAG Pipeline
# -----------------------

def nutrition_agent(query):

    results = vector_store.query(
        query_text=query,
        n_results=2
    )

    response = llm.generate(results, query)

    return response


# -----------------------
# Streaming Function
# -----------------------

def stream_text(text):

    words = text.split(" ")

    partial = ""
    for word in words:
        partial += word + " "
        yield partial
        time.sleep(0.02)


# -----------------------
# Page Config
# -----------------------

st.set_page_config(
    page_title="AI Nutrition Planner",
    page_icon="🥗",
    layout="centered"
)

st.title("🥗 AI Nutrition Planner")
st.caption("Ask for meals, calories, and nutrient breakdowns.")


# -----------------------
# Chat Memory
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------
# Display Previous Messages
# -----------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------
# Chat Input
# -----------------------

prompt = st.chat_input("Ask a nutrition question...")

if prompt:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)


    # Generate assistant response
    with st.chat_message("assistant"):

        placeholder = st.empty()

        with st.spinner("Thinking..."):
            response = nutrition_agent(prompt)

        streamed_text = ""

        for chunk in stream_text(response):
            streamed_text = chunk
            placeholder.markdown(streamed_text)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })