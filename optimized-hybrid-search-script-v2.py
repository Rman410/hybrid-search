# Optimized Hybrid Search Script

import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import ollama
from requests.exceptions import RequestException
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import multiprocessing
from functools import partial
import logging
from sentence_transformers.quantization import quantize_embeddings
from tqdm import tqdm
import re
import asyncio
import faiss
import mmap
from typing import List, Tuple, Dict, Any
from collections import OrderedDict

# Global variables and configuration
WEIGHT_SEMANTIC = 0.7  # Initial weight for semantic search
MAX_CACHE_SIZE = 1000  # Maximum number of items in LRU cache
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2 model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up the Ollama client with custom timeout
client = ollama.Client(timeout=180)  # 3 minutes timeout

# LRU Cache implementation
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Any:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Initialize LRU cache
lru_cache = LRUCache(MAX_CACHE_SIZE)

# Asynchronous function to preload the model
async def preload_model(model_name: str) -> None:
    try:
        await asyncio.to_thread(client.chat, model=model_name, messages=[])
        logger.info(f"Model {model_name} preloaded successfully.")
    except Exception as e:
        logger.error(f"Error preloading model {model_name}: {e}")

# Function to load and preprocess documents using multiprocessing
def load_documents(file_paths: List[str], max_length: int = 512) -> List[str]:
    with multiprocessing.Pool() as pool:
        documents = pool.map(partial(process_file, max_length=max_length), file_paths)
    documents = [doc for file_docs in documents for doc in file_docs]
    logger.info(f"Total document chunks loaded: {len(documents)}")
    return documents

def process_file(file_path: str, max_length: int) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        chunks = content.split('\n\n')
        processed_chunks = []
        for chunk in chunks:
            processed_chunk = preprocess_chunk(chunk)
            if processed_chunk:
                processed_chunks.extend(split_long_text(processed_chunk, max_length))
    logger.info(f"Loaded document: {file_path}")
    return processed_chunks

# Improved text preprocessing function
def preprocess_text(text: str) -> List[str]:
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.isalnum() and token not in stop_words]

def preprocess_chunk(chunk: str) -> str:
    # Remove excessive whitespace and non-printable characters
    return re.sub(r'\s+', ' ', ''.join(char for char in chunk if char.isprintable())).strip()

def split_long_text(text: str, max_length: int = 512) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Function to generate and cache embeddings using memory-mapped files
def generate_and_cache_embeddings(model: SentenceTransformer, documents: List[str], cache_file: str) -> np.ndarray:
    if os.path.exists(cache_file):
        logger.info("Loading cached embeddings...")
        with open(cache_file, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0)
            embeddings = np.frombuffer(mm, dtype=np.float32).reshape(-1, EMBEDDING_DIMENSION)
            if len(embeddings) == len(documents):
                return embeddings
            logger.info("Cached embeddings do not match current documents. Regenerating...")

    logger.info("Generating embeddings...")
    embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=True)
    embeddings_np = embeddings.cpu().numpy()

    with open(cache_file, 'wb') as f:
        f.write(embeddings_np.tobytes())

    return embeddings_np

# Implement Approximate Nearest Neighbor search using FAISS
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(embeddings)
    return index

# Updated hybrid_search function with FAISS
def hybrid_search(query: str, model: SentenceTransformer, documents: List[str], faiss_index: faiss.IndexFlatIP, bm25: BM25Okapi, top_k: int = 5) -> List[Tuple[int, float]]:
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Semantic search with FAISS
    D, I = faiss_index.search(query_embedding.unsqueeze(0).cpu().numpy(), top_k * 2)
    semantic_scores = torch.tensor(D[0])
    semantic_indices = torch.tensor(I[0])
    
    # BM25 search
    tokenized_query = preprocess_text(query)
    bm25_scores = torch.tensor(bm25.get_scores(tokenized_query))
    
    # Combine results
    bm25_scores_rescored = bm25_scores[semantic_indices]
    
    # Normalize scores
    semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
    bm25_scores_rescored = (bm25_scores_rescored - bm25_scores_rescored.min()) / (bm25_scores_rescored.max() - bm25_scores_rescored.min() + 1e-8)
    
    # Automatic balancing: use the harmonic mean of semantic and lexical scores
    combined_scores = 2 / (1/semantic_scores + 1/bm25_scores_rescored)
    
    top_indices = semantic_indices[torch.argsort(combined_scores, descending=True)][:top_k]
    top_scores = combined_scores[torch.argsort(combined_scores, descending=True)][:top_k]
    
    return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_scores)]

def adjust_weight(feedback: str) -> None:
    global WEIGHT_SEMANTIC
    if feedback == 'more_semantic':
        WEIGHT_SEMANTIC = min(1.0, WEIGHT_SEMANTIC + 0.1)
    elif feedback == 'more_lexical':
        WEIGHT_SEMANTIC = max(0.0, WEIGHT_SEMANTIC - 0.1)
    logger.info(f"Adjusted weight_semantic to {WEIGHT_SEMANTIC}")

# Asynchronous function to interact with Ollama
async def ollama_process(query: str, context: str) -> str:
    cached_response = lru_cache.get(f"{query}:{context}")
    if cached_response:
        return cached_response

    try:
        response = await asyncio.to_thread(
            client.chat,
            model="phi3:latest",
            messages=[
                {"role": "system", "content": "You are a helpful assistant tasked to answer user questions about the company Practicing Musician. Use the provided context to answer questions accurately and comprehensively. Provide detailed explanations and examples where appropriate."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nPlease provide a detailed and comprehensive answer:"}
            ],
            options={
                "num_ctx": 4096,
                "temperature": 0.3
            }
        )
        result = response['message']['content']
        lru_cache.put(f"{query}:{context}", result)
        return result
    except RequestException as e:
        if "503" in str(e):
            logger.warning("The server is currently overloaded. Please try again later.")
            return "The server is currently overloaded. Please try again later."
        else:
            logger.error(f"An error occurred: {str(e)}")
            return f"An error occurred: {str(e)}"

def get_user_query() -> str:
    return input("Enter your question (or 'quit' to exit): ")

# Main function to demonstrate hybrid search with Ollama
async def main() -> None:
    # Preload the model
    await preload_model("phi3:latest")
    
    # Load documents
    file_paths = ['C:/Users/roary/OneDrive/Desktop/Files/Practicing Musician/0 - AI and Director Learning/Projects/Ollama/PM Current Answers to FAQs.txt', 'C:/Users/roary/OneDrive/Desktop/Files/Practicing Musician/0 - AI and Director Learning/Projects/Ollama/PM Information on the Website.txt']
    documents = load_documents(file_paths)

    if len(documents) < 5:
        logger.warning("Not enough document chunks for meaningful search. Please check your input files.")
        return

    logger.info(f"Number of documents: {len(documents)}")

    # Initialize the model (using a more suitable pre-trained model for asymmetric search)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate and cache embeddings
    cache_file = 'document_embeddings.mmap'
    document_embeddings = generate_and_cache_embeddings(model, documents, cache_file)
    
    # Build FAISS index
    faiss_index = build_faiss_index(document_embeddings)
    
    # Prepare BM25
    tokenized_corpus = [preprocess_text(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    logger.info("Hybrid search system is ready. You can start asking questions.")
    
    while True:
        query = get_user_query()
        if query.lower() == 'quit':
            logger.info("Thank you for using the Practicing Musician hybrid search system. Goodbye!")
            break
        
        try:
            logger.info("Performing hybrid search...")
            results = hybrid_search(query, model, documents, faiss_index, bm25)
        except RuntimeError as e:
            logger.error(f"An error occurred during search: {e}")
            results = []
        
        if not results:
            logger.warning("No relevant results found. Please try a different query.")
            continue

        logger.info("\nTop search results:")
        for idx, (doc_idx, score) in enumerate(results, 1):
            logger.info(f"Result {idx}:")
            logger.info(f"  Document chunk {doc_idx + 1}")
            logger.info(f"  Relevance Score: {score:.4f}")
            logger.info(f"  Content: {documents[doc_idx][:200]}...")  # Print first 200 characters of the chunk
            logger.info("")

        # Use Ollama to process the top results
        top_context = "\n".join([documents[idx] for idx, _ in results[:3]])
        logger.info("Generating comprehensive answer using Ollama...")
        ollama_response = await ollama_process(query, top_context)
        
        logger.info("\nComprehensive Answer:")
        logger.info(ollama_response)
        logger.info("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
