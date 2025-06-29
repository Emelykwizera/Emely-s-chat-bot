import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Configuration
HANDBOOK_FILE = "Kepler_college_Student_Handbook.pdf"  # Put your PDF in the same folder
VECTOR_STORE_DIR = "kepler_faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free sentence embedding model

# Streamlit app setup
st.set_page_config(page_title="Kepler College Handbook Chatbot", page_icon="📚")
st.title("Kepler College Handbook Chatbot")
st.markdown("Ask questions about the student handbook")

@st.cache_resource
def initialize_chatbot():
    """Initialize all components using only free technologies"""
    # 1. Load and process the PDF
    if not os.path.exists(HANDBOOK_FILE):
        st.error(f"Handbook file '{HANDBOOK_FILE}' not found!")
        return None
    
    loader = PyPDFLoader(HANDBOOK_FILE)
    pages = loader.load()
    
    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    
    # 3. Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if not os.path.exists(VECTOR_STORE_DIR):
        with st.spinner("Creating knowledge base (first time only)..."):
            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store.save_local(VECTOR_STORE_DIR)
    else:
        vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings)
    
    # 4. Set up local LLM with Ollama (free)
    llm = Ollama(
        model="zephyr",  # Free model (run 'ollama pull zephyr' first)
        temperature=0.3
    )
    
    # 5. Create prompt template for paragraph answers
    prompt_template = """You are a helpful assistant for Kepler College students.
    Answer the question in a complete paragraph using only this context:
    
    {context}
    
    Question: {question}
    
    Paragraph answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # 6. Create the QA system
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )
    
    return qa_chain

# Initialize the chatbot
chatbot = initialize_chatbot()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about the handbook..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chatbot({"query": prompt})["result"]
                st.markdown(response)
            except Exception as e:
                st.error("Error generating response")
                response = "Sorry, I couldn't answer that. Please try another question."
                st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})