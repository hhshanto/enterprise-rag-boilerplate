# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Initial project structure for Enterprise-RAG-Boilerplate
- Data ingestion functionality:
  - Implemented download and storage of datasets from Hugging Face
  - Added support for question-answer dataset (subset of SQuAD)
  - Added support for text corpus dataset (tiny_shakespeare)
  - Implemented dataset verification after download
- Embedding generation:
  - Set up embedding generation using Sentence Transformers
  - Implemented batch processing for efficient embedding of text corpus
- Vector database integration:
  - Integrated ChromaDB for storing and managing embeddings
- Testing:
  - Implemented basic unit tests for data ingestion process
- Environment configuration:
  - Added .env file support for secure storage of API keys and endpoints
- Project organization:
  - Established directory structure: data/, data_ingestion/, embedding/, tests/, deployment/, monitoring/
  - Created main.py as the entry point
  - Added requirements.txt for dependency management

### In Progress
- Optimization of embedding generation process
- Expansion of test coverage

### TODO
- Implement retrieval mechanism:
  - Develop query processing system
  - Implement vector similarity search using ChromaDB
  - Create ranking system for retrieved documents
  - Integrate retrieval system with existing data and embedding pipeline
- Implement comprehensive error handling and logging
- Develop deployment scripts and configurations
- Create user documentation and API references
- Set up CI/CD pipeline
- Implement security best practices for handling sensitive information
- Integrate monitoring system

## [0.1.0] -

### Added
- Initial commit
- README.md with project overview
- Basic project structure and essential files

### Security
- Implemented use of .env file for secure storage of API keys