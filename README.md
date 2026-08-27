# Zycus AI Assessment

AI Engineer - Product Support Intern Assessment

## Project Overview

This project implements the solution for the Zycus AI Engineer assessment.

The project includes:

- Customer account data
- Support ticket data
- Product knowledge base
- Knowledge-base indexing
- TF-IDF based retrieval
- RAG pipeline

## Project Structure

```text
Zyncus/
├── data/
│   ├── accounts.json
│   └── tickets.json
├── index/
│   ├── kb_chunks.json
│   ├── tfidf_matrix.pkl
│   └── tfidf_vectorizer.pkl
├── knowledge-base/
│   └── products/
│       └── databridge-pro.md
├── src/
│   ├── data_loader.py
│   ├── kb_index.py
│   ├── kb_index_tfidf.py
│   └── rag_pipeline.py
└── README.md