import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="PDF Chatbot (HuggingFace)", layout="wide")
st.title("🤖 RAG Chatbot with HuggingFace + LangChain")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Load and chunk PDF
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # Embeddings & FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embeddings)
        retriever = vectordb.as_retriever()

        # HuggingFace model
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # RAG QA Chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

        st.success("✅ PDF processed. Ask your questions below:")

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Input box
        user_query = st.chat_input("Ask a question about your PDF:")

        if user_query:
            # Process query
            with st.spinner("Thinking..."):
                response = qa_chain.invoke(user_query)
                answer = response.get("result", response)  # Extract only the answer

            # Show assistant response only (clean)
            st.chat_message("assistant").markdown(f"**{answer}**")

            # Save to history
            st.session_state.chat_history.append(("assistant", f"**{answer}**"))

        # Show conversation history (assistant-only)
        for role, message in st.session_state.chat_history:
            if role == "assistant":
                st.chat_message(role).markdown(message)
