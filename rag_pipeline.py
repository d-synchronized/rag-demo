import chromadb
from chromadb.utils import embedding_functions
import chromadb
from chromadb.utils import embedding_functions
import hashlib

class RAGDemo:
    def __init__(self):
        # Initialize ChromaDB client (persistent)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Use a simple embedding model from sentence-transformers
        # This will download the model if not present
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get a collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_demo_collection",
            embedding_function=self.embedding_fn
        )

    def add_documents(self, documents):
        """
        Adds a list of text documents to the vector database.
        """
        ids = [hashlib.md5(doc.encode()).hexdigest() for doc in documents]
        self.collection.upsert(
            documents=documents,
            ids=ids
        )
        print(f"Added {len(documents)} documents to the database.")

    def query(self, question, n_results=2):
        """
        Retrieves relevant documents and generates an answer (mocked).
        """
        # 1. Retrieve relevant documents
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        retrieved_docs = results['documents'][0]
        
        print(f"\nQuestion: {question}")
        print("Retrieved Context:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. {doc}")
            
        # 2. Generate Answer (Mocked for this demo)
        # In a real scenario, you would pass 'question' and 'retrieved_docs' to an LLM like GPT-4 or Gemini.
        answer = self._mock_generate(question, retrieved_docs)
        
        return answer

    def inspect_collection(self):
        """
        View the content of the vector database.
        """
        print("\n--- Vector Database Content ---")
        # get() fetches all data in the collection
        data = self.collection.get()
        
        ids = data['ids']
        docs = data['documents']
        
        for i in range(len(ids)):
            print(f"ID: {ids[i]}")
            print(f"Document: {docs[i]}")
            print("-" * 30)

    def _mock_generate(self, question, context):
        """
        A placeholder for the LLM generation step.
        """
        # Simple heuristic to make the mock answer look somewhat relevant
        context_str = " ".join(context)
        if "Python" in context_str:
            topic = "Python programming"
        elif "RAG" in context_str:
            topic = "RAG systems"
        else:
            topic = "the provided context"
            
        return f"Based on {topic}, here is a generated answer to '{question}'. (This is a mock response using the retrieved context)."

def main():
    # Sample documents
    documents = [
        "Retrieval-Augmented Generation (RAG) enhances LLM accuracy by retrieving external data.",
        "ChromaDB is an open-source vector database for building AI applications.",
        "Python is a popular programming language for data science and AI.",
        "The capital of France is Paris.",
        "Machine learning models require data for training."
    ]
    
    rag = RAGDemo()
    
    # Index documents
    rag.add_documents(documents)
    
    # Test queries
    q1 = "What is RAG?"
    answer1 = rag.query(q1)
    print(f"Answer: {answer1}\n")
    
    q2 = "Tell me about ChromaDB"
    answer2 = rag.query(q2)
    print(f"Answer: {answer2}\n")

    # View database content
    rag.inspect_collection()

if __name__ == "__main__":
    main()
