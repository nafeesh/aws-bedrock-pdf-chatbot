# AWS Bedrock PDF Chatbot

This project is a **PDF chatbot** built with **Streamlit** and **AWS Bedrock**. 
It allows users to upload PDF documents and ask questions interactively. The chatbot 
retrieves relevant information from the PDFs using **Retrieval-Augmented Generation (RAG)** 
and provides concise, context-aware answers.

## Features
- Upload multiple PDFs for interactive Q&A.
- Powered by **AWS Bedrock LLMs** for fast and accurate responses.
- Retrieval-Augmented Generation (RAG) using FAISS vector store.
- Handles follow-up questions and maintains conversation context.
- Includes a legal disclaimer for safe and responsible usage.
- Streamlit UI for an intuitive chat experience.

## Tech Stack
- **Frontend:** Streamlit
- **Backend / LLM:** AWS Bedrock (ChatBedrockConverse, BedrockEmbeddings)
- **Vector Database:** FAISS
- **Text Splitting:** langchain_text_splitters
- **Document Loader:** langchain_community.document_loaders

## Disclaimer
This chatbot provides AI-generated answers based on the content of your PDFs. 
Responses may be incomplete, inaccurate, or biased. Do not upload confidential 
or sensitive information. Always verify answers before making decisions.

