import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

st.set_page_config(page_title="ChromaDB Viewer", layout="wide")

st.title("🔍 ChromaDB Viewer")

# Initialize Client
@st.cache_resource
def get_client():
    return chromadb.PersistentClient(path="./chroma_db")

client = get_client()

# Initialize Embedding Function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Get Collection
try:
    collection = client.get_collection(
        name="rag_demo_collection",
        embedding_function=embedding_fn
    )
    st.success("Connected to `rag_demo_collection`")
except Exception as e:
    st.error(f"Could not find collection: {e}")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["Browse Documents", "Test Retrieval"])

with tab1:
    st.header("Stored Documents")
    
    # Get all documents
    data = collection.get()
    
    if data['ids']:
        df = pd.DataFrame({
            'ID': data['ids'],
            'Document': data['documents'],
            'Metadata': [str(m) for m in data['metadatas']] if data['metadatas'] else ["None"] * len(data['ids'])
        })
        st.dataframe(df, use_container_width=True)
        st.caption(f"Total Documents: {len(data['ids'])}")
    else:
        st.info("No documents found in the collection.")

with tab2:
    st.header("Test Retrieval")
    
    query = st.text_input("Enter a query:", "What is RAG?")
    n_results = st.slider("Number of results:", 1, 10, 2)
    
    if st.button("Search"):
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        st.subheader("Results")
        for i, doc in enumerate(results['documents'][0]):
            st.markdown(f"**{i+1}.** {doc}")
            st.divider()
