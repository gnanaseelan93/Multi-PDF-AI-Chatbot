import streamlit as st
import os
import tempfile
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
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

def save_faiss_index(db, path=FAISS_INDEX_PATH):
    """Save FAISS index to a directory."""
    db.save_local(path)

def load_faiss_index(path=FAISS_INDEX_PATH):
    """Load FAISS index with deserialization enabled safely."""
    if os.path.exists(path):
        try:
            db = FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            
            # ✅ Debugging step: Print the number of documents in FAISS
            num_docs = len(db.docstore._dict)
            st.write(f"✅ FAISS index loaded with {num_docs} document chunks.")
            
            return db
        except Exception as e:
            st.error(f"❌ Failed to load FAISS index: {e}")
    return None

def process_pdfs(pdf_files):
    """Process multiple PDFs and update FAISS vector store."""
    all_documents = []

    # ✅ Step 1: Clear old FAISS index when processing new PDFs
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)  # Completely remove old FAISS index

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
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        all_documents.extend(text_splitter.split_documents(docs))
        os.remove(temp_pdf_path)
    
    if not all_documents:
        return None
    
    # ✅ Step 2: Create a NEW FAISS index from scratch
    db = FAISS.from_documents(all_documents, OpenAIEmbeddings())
    save_faiss_index(db)
    return db

def setup_chain(db):
    """Setup retrieval chain using FAISS and OpenAI LLM."""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant answering questions based on multiple uploaded documents.
    - Use ONLY the provided context to answer the question.
    - If relevant information is found in multiple places, merge them into a single coherent answer.
    - If a question asks about multiple topics, answer them separately.
    <context>
    {context}
    </context>
    Question: {input}""")
    
    documents_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={"k": 8})
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)
    
    return retrieval_chain

# Streamlit UI Setup
st.set_page_config(page_title="Multi-PDF Q&A Chatbot", page_icon="📄", layout="wide")
st.sidebar.title("⚙️ Settings")

# ✅ Button to CLEAR FAISS and force re-upload
if st.sidebar.button("🗑 Clear FAISS Index and Rebuild"):
    shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
    st.success("✅ FAISS index cleared. Re-upload PDFs.")

db = load_faiss_index()

st.title("📄 Multi-PDF Q&A Chatbot")
st.markdown("Interact with multiple PDFs using AI-powered search!")

uploaded_files = st.file_uploader("📤 Upload PDFs", type=["pdf"], accept_multiple_files=True, help="Upload one or more PDFs to start querying")

# ✅ Step 3: Remove a PDF and update FAISS
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        db = process_pdfs(uploaded_files)  # Process only the selected PDFs
    if db:
        st.success("✅ PDFs processed successfully!")
    else:
        st.error("❌ Failed to process PDFs.")

if db:
    retrieval_chain = setup_chain(db)
    
    st.markdown("### 📝 Ask Multiple Questions")
    query = st.text_area("🔍 Type your questions here (you can ask multiple questions):")

    if st.button("💡 Get Answer"):
        if query.strip() == "":
            st.warning("⚠️ Please enter at least one question before submitting.")
        else:
            response = retrieval_chain.invoke({"input": query})
            st.success(response['answer'])
