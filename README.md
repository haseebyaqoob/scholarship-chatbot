# NED University Scholarship Assistant

A conversational assistant built to help NED University students find and understand scholarships. Students can ask questions in plain English and get answers about available scholarships, eligibility, deadlines, funding types, and application details.

---

## What It Does

- Answers questions about scholarships using structured CSV data and detailed text documents
- Lets students filter scholarships by GPA, domicile, level of study, field, and funding type
- Recommends scholarships based on a student's profile (GPA, field, province, etc.)
- Tracks a student's profile across the conversation to give personalised suggestions
- Handles counting, sorting, and comparison queries (e.g. "which scholarship has the lowest GPA requirement")

---

## Who It Is For

Students at NED University of Engineering and Technology, Karachi, who want a quick way to explore scholarship options without manually going through spreadsheets or PDFs.

---

## How It Works

The system has three main components:

**1. Scholarship Index**
Loads scholarship data from a CSV file and optional per-scholarship text files stored in a local folder. The CSV holds structured metadata (deadlines, amounts, GPA requirements, etc.) and the text files hold detailed narrative information (eligibility criteria, documents required, how to apply, bond conditions, etc.).

**2. Router**
Classifies each user query into one of seven intents using a two-stage approach:
- Stage 1 identifies what the user wants (a structured data query, detailed info about a specific scholarship, a general recommendation, a profile update, a greeting, etc.)
- Stage 2 extracts structured parameters only when needed (filters and sort specs for CSV queries, or profile fields for profile updates)

This split keeps each LLM call small and focused, which improves reliability on smaller models.

**3. RAG Engine**
For discovery and recommendation queries, relevant scholarship content is retrieved using hybrid search. It combines semantic search (FAISS with sentence-transformer embeddings) and keyword search (BM25) and scores results as a weighted mix of both. Only chunks above a minimum relevance threshold are passed to the LLM for answer generation.

---

## Techniques Used

- Sentence-transformer embeddings (semantic similarity)
- FAISS vector index (fast nearest-neighbour search)
- BM25 (keyword-based retrieval via rank-bm25)
- Hybrid retrieval scoring (0.6 semantic + 0.4 BM25, with score threshold filtering)
- Two-stage LLM routing (intent classification then schema extraction)
- Fuzzy name matching with rapidfuzz (handles typos and partial names)
- Auto-expiry of scholarships past their deadline
- Config-driven design: no hardcoded scholarship names, column names, or funding categories

---

## Adding a New Scholarship

Drop a new row into the CSV file and optionally add a text file with detailed information. No code changes needed.

---

## Requirements

- Python 3.10+
- Ollama (local LLM server)
- Dependencies: faiss-cpu, sentence-transformers, rank-bm25, rapidfuzz, pandas, PyYAML
