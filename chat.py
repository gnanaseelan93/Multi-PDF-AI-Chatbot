import streamlit as st
import os
import tempfile
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables securely
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ Missing OPENAI_API_KEY. Please check your .env file.")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key

FAISS_INDEX_PATH = "faiss_index"
UPLOADED_FILES_KEY = "uploaded_files"

# Ensure `processing` is initialized in session state
if "processing" not in st.session_state:
    st.session_state.processing = False

# Utility functions for FAISS Index
def save_faiss_index(db, path=FAISS_INDEX_PATH):
    db.save_local(path)

def load_faiss_index(path=FAISS_INDEX_PATH):
    if os.path.exists(path):
        try:
            db = FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            num_docs = len(db.docstore._dict)
            st.write(f"✅ FAISS index loaded with {num_docs} document chunks.")
            return db
        except Exception as e:
            st.error(f"❌ Failed to load FAISS index: {e}")
    return None

def delete_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)

# PDF Processing Function
def process_pdfs(pdf_files):
    all_documents = []
    
    # If no files are uploaded, delete index
    if not pdf_files:
        delete_faiss_index()
        st.session_state[UPLOADED_FILES_KEY] = []
        return None

    # Remove old FAISS index to avoid stale data
    delete_faiss_index()
    
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.read())
            temp_pdf_path = temp_pdf.name
        try:
            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
        except Exception as e:
            st.error(f"❌ Error loading PDF {pdf_file.name}: {e}")
            os.remove(temp_pdf_path)
            continue
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        all_documents.extend(text_splitter.split_documents(docs))
        os.remove(temp_pdf_path)
    
    if not all_documents:
        return None

    db = FAISS.from_documents(all_documents, OpenAIEmbeddings())  
    save_faiss_index(db)
    
    return db

# Retrieval Chain Setup
def setup_chain(db):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")


    documents_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={"k": 2, "score_threshold": 0.5})
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)
    
    return retrieval_chain

# Streamlit UI Setup
st.set_page_config(page_title="Multi-PDF Q&A Chatbot", page_icon="📄", layout="wide")
st.sidebar.title("⚙️ Settings")

# Load previously uploaded files
if UPLOADED_FILES_KEY not in st.session_state:
    st.session_state[UPLOADED_FILES_KEY] = []

# Clear FAISS Index
if st.sidebar.button("🗑 Clear FAISS Index and Rebuild"):
    delete_faiss_index()
    st.session_state[UPLOADED_FILES_KEY] = []
    st.success("✅ FAISS index cleared. Re-upload PDFs.")

db = load_faiss_index()

st.title("📄 Multi-PDF Q&A Chatbot")
st.markdown("Interact with multiple PDFs using AI-powered search!")

uploaded_files = st.file_uploader("📤 Upload PDFs", type=["pdf"], accept_multiple_files=True, help="Upload one or more PDFs to start querying")

# Detect file removals or additions
uploaded_filenames = [file.name for file in uploaded_files] if uploaded_files else []
previously_uploaded = st.session_state[UPLOADED_FILES_KEY]

if uploaded_filenames != previously_uploaded:
    st.session_state[UPLOADED_FILES_KEY] = uploaded_filenames
    st.session_state.processing = True  # ✅ Already initialized, no error

    with st.spinner("Processing PDFs..."):
        db = process_pdfs(uploaded_files)

    if db:
        st.success("✅ PDFs processed successfully!")
    else:
        st.warning("⚠️ No valid PDFs uploaded. Index cleared.")

    st.session_state.processing = False  # ✅ Already initialized, no error

if db:
    retrieval_chain = setup_chain(db)
    
    query = st.text_input("🔍 Type your questions here:", disabled=st.session_state.processing)
    get_answer_btn = st.button("💡 Get Answer", disabled=st.session_state.processing)

    if get_answer_btn:
        if query.strip() == "":
            st.warning("⚠️ Please enter a question before submitting.")
        elif query.count("?") > 1:  # Basic check for multiple questions
            st.warning("⚠️ Please ask only one question at a time.")
        else:
            with st.spinner("Processing Query... Please wait."):
                response = retrieval_chain.invoke({"input": query})
            st.success(response['answer'])
