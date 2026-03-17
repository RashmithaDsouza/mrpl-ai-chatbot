import os
import streamlit as st
import fitz

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# ---------------- CONFIG ----------------

UPLOAD_DIR = "uploads"
VECTOR_DIR = "vector_store"
LOGO_PATH = "assets/logo1.png"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

st.set_page_config(page_title="MRPL AI Assistant", layout="wide")


# ---------------- EMBEDDINGS ----------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()


# ---------------- PDF TEXT EXTRACTION ----------------

def extract_text(pdf_path):

    docs = []

    with fitz.open(pdf_path) as pdf:

        for page in pdf:

            text = page.get_text()

            if text:
                docs.append(
                    Document(page_content=text)
                )

    return docs


# ---------------- CREATE VECTOR STORE ----------------

def build_vector_store(files):

    all_docs = []

    for file in files:

        path = os.path.join(UPLOAD_DIR, file.name)

        if not os.path.exists(path):

            with open(path, "wb") as f:
                f.write(file.getbuffer())

        docs = extract_text(path)

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    db = FAISS.from_documents(chunks, embeddings)

    return db


# ---------------- QA CHAIN ----------------

def get_chain(db):

    llm = Ollama(
        model="phi3",
        base_url="http://localhost:11434",
        temperature=0
    )

    prompt = PromptTemplate(
        template="""
You are analyzing MRPL annual reports.

Answer using ONLY the provided context.

If the question asks about financial values like:
revenue, profit, PAT, turnover etc.

Return only the number and unit.

Example:
Revenue: ₹12,47,360.30 million

If not found say:
Data not found in report.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context","question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k":5}),
        chain_type_kwargs={"prompt":prompt}
    )


# ---------------- SESSION ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None


# ---------------- SIDEBAR ----------------

with st.sidebar:

    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)

    st.title("MRPL AI Assistant")

    if st.button("New Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.subheader("Uploaded Reports")

    files = os.listdir(UPLOAD_DIR)

    if len(files) == 0:
        st.write("No reports uploaded")

    for f in files:
        st.write("📄", f.replace(".pdf",""))


# ---------------- MAIN PAGE ----------------

st.title("MRPL Financial Chatbot")

uploaded_files = st.file_uploader(
    "Upload Annual Reports",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    with st.spinner("Processing reports..."):

        db = build_vector_store(uploaded_files)

        st.session_state.chain = get_chain(db)

    st.success("Reports ready for questions")


# ---------------- CHAT DISPLAY ----------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


question = st.chat_input("Ask about MRPL reports")


# ---------------- PROCESS QUESTION ----------------

if question:

    st.session_state.messages.append(
        {"role":"user","content":question}
    )

    with st.chat_message("user"):
        st.write(question)

    if st.session_state.chain is None:

        answer = "Please upload reports first."

    else:

        with st.spinner("Analyzing reports..."):

            try:

                result = st.session_state.chain.invoke(
                    {"query":question}
                )

                answer = result["result"]

            except:

                answer = "⚠ Start Ollama first: ollama serve"

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )
