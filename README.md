# 💊 Drug Recommendation System

This project is a drug information retrieval system developed in Python. It uses semantic search to find drugs and drug reviews that are relevant to a user's natural-language query.

The user can enter a description such as:

"I want something for depression"

and the system searches a collection of drug reviews stored in ChromaDB. The most relevant results are then displayed through a simple Gradio web interface. The user can also select a specific condition and choose the number of results to display.

The main technologies used in the project are **Python, Pandas, LangChain, ChromaDB, Hugging Face Sentence Transformers, and Gradio**.

## How the System Works

The system follows a simple retrieval pipeline:

```text
User Query
    ↓
Sentence Transformer
    ↓
Text Embedding
    ↓
ChromaDB Semantic Search
    ↓
Relevant Drug Reviews
    ↓
Condition Filtering
    ↓
Ranking
    ↓
Gradio Results

Feel free to play with it !! 