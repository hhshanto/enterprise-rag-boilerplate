# Data Embedding Tests

This document outlines the tests performed in `test_data_embedding.py` for the Enterprise-RAG-Boilerplate project.

## Overview

The `test_data_embedding.py` file contains unit tests for the `DataEmbedding` class, which is responsible for generating and storing embeddings for the text corpus used in our RAG system.

## Test Cases

### 1. Test Initialization

**Method**: `test_initialization`

This test ensures that the `DataEmbedding` class initializes correctly:
- Verifies that the Hugging Face token is set
- Checks if the ChromaDB client is properly initialized
- Confirms that the embedding function is set up correctly
- Validates that the correct embedding model type is being used

### 2. Test Directory Structure

**Method**: `test_directory_structure`

This test checks if all required directories are created and exist:
- Verifies the existence of the root directory
- Checks for the presence of the data directory
- Ensures the embedding directory is created

### 3. Test Data Loading

**Method**: `test_data_loading`

This test verifies that the text corpus can be loaded correctly:
- Attempts to load the text dataset from disk
- Checks if the loaded dataset is not None
- Ensures that the dataset contains data (length > 0)

### 4. Test Collection Creation

**Method**: `test_collection_creation`

This test checks if the ChromaDB collection is created correctly:
- Verifies that a collection named "text_embeddings" exists in ChromaDB

### 5. Test Embedding Storage

**Method**: `test_embedding_storage`

This test ensures that embeddings are stored correctly in the ChromaDB collection:
- Retrieves the "text_embeddings" collection
- Checks if the collection contains embeddings (count > 0)

### 6. Test Metadata Structure

**Method**: `test_metadata_structure`

This test verifies the correct structure of metadata stored with embeddings:
- Performs a query to retrieve a sample embedding
- Checks if metadata is present
- Verifies that the 'source' field in metadata is set to "shakespeare"

### 7. Test Batch Processing

**Method**: `test_batch_processing`

This test checks if batch processing of queries works correctly:
- Performs multiple queries at once
- Verifies that the correct number of results is returned
- Ensures that all expected result fields (ids, documents, metadatas) are present

### 8. Test ID Format

**Method**: `test_id_format`

This test verifies the format of document IDs in the ChromaDB collection:
- Retrieves a sample document
- Checks if the ID follows the expected format (e.g., "text_X" where X is a number)

## Setup and Teardown

- The test suite uses `setUpClass` to initialize the `DataEmbedding` instance and embed the text corpus before running tests.
- It cleans up any existing embedding directory before starting to ensure a fresh test environment.

## Dependencies

- unittest
- chromadb
- dotenv
- datasets (Hugging Face)

## Note

These tests ensure the robustness and correctness of the embedding generation and storage process, which is crucial for the performance of the RAG system.