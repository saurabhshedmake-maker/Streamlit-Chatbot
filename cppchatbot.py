import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


st.set_page_config(
    page_title="C++ AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 C++ AI Assistant")
st.write("Ask any C++ question based on your knowledge base.")


@st.cache_resource
def load_vector_database():
    # Load Text File
    loader = TextLoader("cppTextData.txt", encoding="utf-8")
    documents = loader.load()

    # Split Documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    docs = splitter.split_documents(documents)

    # Create Embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Create Vector Store
    db = FAISS.from_documents(docs, embedding)

    return db


with st.spinner("Loading Knowledge Base..."):
    db = load_vector_database()

st.success("Knowledge Base Loaded Successfully ✅")

query = st.text_input(
    "Ask your C++ Question",
    placeholder="Example: What is polymorphism?"
)

if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    else:

        results = db.similarity_search(query, k=3)

        st.subheader("Answer")

        if len(results) == 0:
            st.error("No relevant answer found.")
        else:
            for i, doc in enumerate(results, start=1):
                st.markdown(f"### Result {i}")
                st.write(doc.page_content)
                st.divider()