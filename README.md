# Enterprise-RAG-Boilerplate

A modular, scalable template for building Retrieval-Augmented Generation (RAG) pipelines for enterprise applications.

## Overview

This project provides a robust framework for implementing RAG pipelines, focusing on:
- Data ingestion and embedding generation
- Vector database integration
- Semantic search capabilities
- Comprehensive testing and evaluation
- Enterprise-grade deployment options

## Project Structure

```plaintext
enterprise-rag-boilerplate/
├── data/
│   ├── ragData/
│   │   ├── text_corpus/      # Shakespeare text dataset
│   │   └── question_answer/  # QA dataset
│   └── embedding/            # ChromaDB embeddings storage
├── embedding/
│   ├── __init__.py
│   └── data_embedding.py     # Core embedding functionality
├── tests/
│   ├── __init__.py
│   ├── test_data_embedding.py
│   └── exploreDataset.py     # Dataset exploration utilities
├── requirements.txt
└── README.md
```

## Features

### 1. Data Embedding Pipeline
- Efficient batch processing of text documents
- Integration with HuggingFace's sentence transformers
- Configurable embedding models and parameters
- Persistent storage using ChromaDB

### 2. Vector Database Integration
- Seamless integration with ChromaDB
- Scalable document storage and retrieval
- Metadata management for document tracking
- Efficient batch operations

### 3. Testing Framework
Comprehensive test suite covering:
- Initialization and configuration
- Data loading and processing
- Embedding generation
- Semantic search functionality
- Error handling and edge cases

## Getting Started

### Prerequisites
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate    # Windows
```

### Installation
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file:
```
HuggingAccessToken=your_huggingface_token
```

### Running Tests
```bash
python -m unittest tests/test_data_embedding.py -v
```

## Core Components

### Data Embedding
```python
from embedding.data_embedding import DataEmbedding

embedder = DataEmbedding()
embedder.embed_text_corpus()
```

### Dataset Exploration
```python
from tests.exploreDataset import print_dataset_analysis

print_dataset_analysis("path/to/dataset", "Dataset Name")
```

## Testing Coverage

1. **Initialization Tests**
   - Environment configuration
   - Model initialization
   - Directory structure

2. **Data Processing Tests**
   - Dataset loading
   - Batch processing
   - Document handling

3. **Embedding Tests**
   - Vector generation
   - Storage verification
   - Dimension validation

4. **Search Tests**
   - Semantic search functionality
   - Result relevance
   - Query handling

5. **System Tests**
   - Error handling
   - Resource cleanup
   - Performance validation

## Development Workflow

1. **Setup Environment**
   - Clone repository
   - Install dependencies
   - Configure environment variables

2. **Data Preparation**
   - Place datasets in `data/ragData/`
   - Run dataset exploration tools
   - Verify data structure

3. **Implementation**
   - Modify embedding parameters
   - Implement custom processing
   - Add new functionality

4. **Testing**
   - Run test suite
   - Verify all components
   - Add new test cases

## Best Practices

1. **Data Management**
   - Use consistent data formats
   - Implement proper error handling
   - Maintain data versioning

2. **Testing**
   - Write comprehensive tests
   - Use meaningful test names
   - Include edge cases

3. **Code Organization**
   - Follow modular design
   - Document functions
   - Use type hints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace for transformer models
- ChromaDB for vector storage
- Shakespeare dataset for testing

---

**Note**: This is a boilerplate project. Customize components based on your specific needs while maintaining the modular structure and testing practices.

For detailed implementation examples and advanced features, check the individual component documentation.