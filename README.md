# Health-and-Fitness-RAG-Chatbot
Multilingual RAG-based chatbot for health and fitness queries using English and French corpora with LaBSE embeddings.

This repository contains a **Retrieval-Augmented Generation (RAG)** chatbot pipeline that supports multilingual document understanding and querying, specifically for English and French PDFs.

---

##  Project Overview

This project enables you to:

-  Merge and clean multilingual PDF documents
-  Insert cleaned data into a vector store for semantic search
-  Query the documents using a chatbot interface powered by RAG

---

## Repository Structure

| File Name              | Description |
|------------------------|-------------|
| `merge_and_clean.py`   | Merges and cleans multilingual PDFs. |
| `insert_rag.py`        | Inserts cleaned document embeddings into the vector store (e.g., Pinecone ). |
| `query_rag_chatbot.py` | Provides a chatbot interface for querying the documents using RAG. |
| `Cleaned_English.pdf`  | Cleaned English document used in the chatbot pipeline. |
| `Cleaned_French.pdf`   | Cleaned French document used in the chatbot pipeline. |

---

Developed by [Waleed Shehzad]. For inquiries or contributions, please open an issue or pull request.
