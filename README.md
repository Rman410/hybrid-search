# Optimized Hybrid Search Script

This Python script implements an advanced hybrid search system that combines semantic and lexical search techniques to process and retrieve information from large text documents. It uses Ollama for generating comprehensive answers based on the retrieved context.

## Features

- Hybrid search combining semantic (FAISS) and lexical (BM25) search
- Document splitting and preprocessing
- Embedding generation and caching
- Asynchronous processing
- LRU caching for Ollama responses
- Dynamic weight adjustment between semantic and lexical search
- Local processing suitable for sensitive or proprietary information

## Requirements

- Python 3.7+
- See requirements.txt for Python package dependencies
- Ollama installed locally (https://ollama.com/)

## Installation

1. Clone this repository
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Ensure Ollama is installed. This version has "phi3:latest" model as the default.

## Usage

1. Place your text documents in the specified directory (replace the paths 'path_to_your_file_1.txt' and 'path_to_your_file_2.txt' with your own .txt document paths).
2. Replace "[company]" in the script for your personalized use case.
3. Run the script:
   ```
   python hybrid-search.py
   ```
4. Enter your questions when prompted
5. Type 'quit' to exit the program

## Note

This script is designed for local processing and may be slower than cloud-based solutions, especially for large document sets. However, it provides a high level of privacy and control over the data processing pipeline.
