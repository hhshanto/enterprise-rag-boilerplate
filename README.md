# Enterprise-RAG-Boilerplate

*A modular, scalable template for building Retrieval-Augmented Generation (RAG) pipelines for enterprise applications.*

## Project Goals

1. **Reusable Framework**: Provide a scalable framework for setting up RAG pipelines.
2. **Enterprise Integration**: Enable seamless integration with enterprise tools like Azure OpenAI, vector databases, and monitoring platforms.
3. **Simplified Operations**: Simplify evaluation, deployment, and monitoring of RAG pipelines.
4. **Best Practices**: Follow industry best practices for modular code, CI/CD, and infrastructure-as-code.

---

## Architecture Overview

### 1. Input Layer
- Accepts user queries via REST API or web interface.
- Preprocesses and reformulates queries for retrieval.

### 2. Retrieval Layer
- Fetches relevant documents or embeddings from a vector database.
- Supports hybrid retrieval (vector and keyword-based).

### 3. LLM Layer
- Passes retrieved context to a Language Model (e.g., Azure OpenAI, Hugging Face models).
- Augments prompts with retrieval results.

### 4. Evaluation Layer
- Evaluates outputs for relevance, factual correctness, and latency.
- Logs metrics for continuous improvement.

### 5. Deployment Layer
- Dockerized deployment for local testing.
- Kubernetes or serverless options for scalability.

### 6. Monitoring Layer
- Observes system performance and logs errors.
- Alerts for anomalies or drift detection.

---

## Project Features

### 1. Data Ingestion and Vector Store Integration
- **Supported Formats**: PDF, Word, JSON, and CSV.
- **Document Processing Pipeline**:
  - Text extraction using tools like `PyPDF2` or `pdfplumber`.
  - Chunking for context windows.
  - Embedding generation using Azure OpenAI or Hugging Face models.
- **Vector Database Setup**:
  - Integrations with Pinecone, Weaviate, or Qdrant.
  - Code examples to insert documents and query embeddings.

---

### 2. Retrieval-Augmented Generation Pipeline
- **Query Preprocessing**:
  - Spell-check, keyword extraction, or query reformulation.
- **Hybrid Retrieval**:
  - Combine vector-based and keyword search methods.
- **Prompt Augmentation**:
  - Include retrieved documents in prompts.
  - Customizable prompt templates for different tasks.

---

### 3. LLM Integration
- **Model Integration Options**:
  - Azure OpenAI endpoints for GPT-based models.
  - Hugging Face transformers for local or open-source models.
- **Prompt Management**:
  - Modular prompt templates for different tasks (e.g., summarization, Q&A).

---

### 4. Evaluation Framework
- **Automated Metrics**:
  - Evaluate outputs for relevance, accuracy, and latency.
  - Use tools like RAGAS or promptfoo for scoring.
- **Human Feedback Loop**:
  - Scripts for collecting human feedback on responses.
- **Drift Detection**:
  - Compare embedding similarity over time to detect drift.

---

### 5. Deployment & Scalability
- **Local Deployment**:
  - Docker Compose setup for local testing.
- **Cloud Deployment**:
  - Templates for provisioning resources on Azure.
  - Kubernetes manifests for container orchestration.
- **Serverless Option**:
  - Examples using Azure Functions or AWS Lambda.

---

### 6. Monitoring and Observability
- **Performance Metrics**:
  - Log response times, token usage, and API costs.
  - Use Prometheus and Grafana for monitoring.
- **Error Handling**:
  - Graceful degradation if retrieval fails.
  - Log errors for debugging.
- **Alerts**:
  - Set up alerts for high latency or token consumption.

---

## Directory Structure

```plaintext
enterprise-rag-boilerplate/
├── data_ingestion/           # Document ingestion and preprocessing
│   ├── extract_text.py       # Extract text from documents
│   ├── chunking.py           # Document chunking logic
│   └── embeddings.py         # Generate embeddings
├── retrieval/                # Retrieval logic
│   ├── vector_search.py      # Vector-based search methods
│   ├── hybrid_search.py      # Combine vector and keyword search
│   └── database.py           # Vector database interactions
├── llm/                      # LLM integration and prompt management
│   ├── prompt_augmentation.py # Prompt construction logic
│   └── llm_api.py            # API calls to Azure or Hugging Face
├── evaluation/               # Evaluation tools
│   ├── metrics.py            # Define evaluation metrics
│   ├── promptfoo_tests.py    # Integration with promptfoo
│   └── human_feedback.py     # Human-in-the-loop evaluation scripts
├── deployment/               # Deployment configuration
│   ├── Dockerfile            # Docker setup for testing
│   ├── k8s.yaml              # Kubernetes manifests
│   └── terraform/            # Infrastructure-as-code templates
├── monitoring/               # Monitoring and observability
│   ├── logging.py            # Logging setup
│   └── alerts.py             # Alert configurations
└── README.md                 # Project documentation