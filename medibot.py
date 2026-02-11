import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
    ChatHuggingFace,
)

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN not found. Please add it to the .env file.")
    st.stop()

# -------------------------------------------------
# CONFIG (SUPPORTED MODEL)
# -------------------------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

# -------------------------------------------------
# Vector Store
# -------------------------------------------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

# -------------------------------------------------
# Prompt
# -------------------------------------------------
def get_prompt():
    template = """
Use the provided context to answer the user's question.
If you do not know the answer, say you do not know.
Do not make up information.

Context: {context}
Question: {question}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# -------------------------------------------------
# Hugging Face Conversational LLM
# -------------------------------------------------
def load_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="conversational",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.3,
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=endpoint)

# -------------------------------------------------
# Streamlit App
# -------------------------------------------------
def main():
    st.title("🩺 MediBot (Hugging Face RAG)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_query = st.chat_input("Ask a medical question")

    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )

        try:
            vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": get_prompt()}
            )

            response = qa_chain.invoke(user_query)

            answer = response["result"]
            sources = response["source_documents"]

            final_answer = answer + "\n\n**Sources:**\n" + str(sources)

            st.chat_message("assistant").markdown(final_answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer}
            )

        except Exception as e:
            st.error(f"ERROR: {e}")

if __name__ == "__main__":
    main()
