# RAG Pipeline Demo

This is a simple demonstration of a Retrieval-Augmented Generation (RAG) pipeline using Python, ChromaDB, and Sentence Transformers.

## Prerequisites

- Python 3.8+

## Installation

1.  Clone or navigate to this directory.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the `rag_pipeline.py` script:

```bash
python rag_pipeline.py
```

## Visualizing the Database

To view the stored documents and test retrieval interactively, run the Streamlit viewer:

```bash
streamlit run db_viewer.py
```

## How it Works

1.  **Initialization**: Sets up an in-memory ChromaDB client and a Sentence Transformer embedding model (`all-MiniLM-L6-v2`).
2.  **Indexing**: Adds a few sample sentences to the vector database. The text is embedded and stored.
3.  **Retrieval**: When a question is asked, the system embeds the question and finds the most similar documents in the database.
4.  **Generation**: (Mocked) In a real system, the retrieved context and question would be sent to an LLM. Here, we print the retrieved context and a placeholder answer.
