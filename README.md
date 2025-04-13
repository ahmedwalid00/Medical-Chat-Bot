# Medical Chatbot using LangChain, LlamaIndex, and ChromaDB

This project implements an intelligent **Medical Chatbot** that answers medical-related questions using a retrieval-augmented generation (RAG) pipeline. It leverages the power of **LangChain**, **LlamaIndex**, **ChromaDB**, and **OpenAI's GPT** to provide accurate and context-aware responses.

## Project Overview

The chatbot is designed to interact in a conversational manner and retrieve relevant information from medical documents. A custom retriever was built to integrate **LlamaIndex** with **LangChain**, enabling the pipeline to understand, fetch, and present relevant data from stored files.

Users can ask health-related queries, and the system will use the provided documents and semantic search to return helpful answers.

## Technologies & Tools Used

- **LangChain**: For building the conversational pipeline and integration with language models.
- **LlamaIndex**: To parse, index, and retrieve chunks of documents using embeddings.
- **ChromaDB**: Vector database used to persist and search document embeddings.
- **HuggingFace Embeddings**: For converting documents into vector representations.
- **OpenAI GPT-4o-mini**: The chat model used for generating answers.
- **Flask (optional for deployment)**: Can be integrated to serve this as a web application.
- **Python**: Core programming language for implementation.
- **dotenv**: For securely managing API keys.

## ✅ Features

- RAG pipeline combining GPT and semantic search.
- Fast and efficient vector search using ChromaDB.
- Custom integration of LlamaIndex inside LangChain’s retriever logic.
- Maintains conversation context using memory buffers.

---


