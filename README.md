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
```
### Example

The following screenshot shows an example of the drug information retrieval system.

The query used in this example was related to **depression** and asked which drugs were the most relevant based on the available reviews in the dataset.


[![image alt](https://github.com/CharalamposZisis/Drug-Recommendation-System/blob/815b106d3d6bd353f72ef9a56d7081fe0b258bfe/example_of_text_classif.png)


Feel free to play with it !! 
