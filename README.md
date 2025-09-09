# Simple RAG API

A simple Python API for RAG (Retrieval-Augmented Generation) operations using FastAPI.

## Features

- **Document Management**: Add, list, retrieve, and delete documents
- **Query Interface**: Query the knowledge base with questions
- **RESTful API**: Clean REST endpoints with automatic OpenAPI documentation
- **CORS Support**: Cross-origin requests enabled
- **Health Checks**: Built-in health check endpoints

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint

### Document Management
- `POST /documents` - Add a new document to the knowledge base
- `GET /documents` - List all documents
- `GET /documents/{doc_id}` - Get a specific document by ID
- `DELETE /documents/{doc_id}` - Delete a document

### Query
- `POST /query` - Query the knowledge base with a question

## Usage Examples

### Add a Document
```bash
curl -X POST "http://localhost:8000/documents" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "Python is a high-level programming language known for its simplicity and readability.",
       "metadata": {"topic": "programming", "language": "python"}
     }'
```

### Query the Knowledge Base
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is Python?"
     }'
```

### List All Documents
```bash
curl -X GET "http://localhost:8000/documents"
```

## API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Development

### Running in Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Project Structure
```
.
├── main.py          # Main FastAPI application
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Notes

- This is a simple implementation using in-memory storage
- For production use, consider integrating with:
  - A proper database (PostgreSQL, MongoDB, etc.)
  - Vector databases for semantic search (Pinecone, Weaviate, etc.)
  - LLM APIs for better answer generation (OpenAI, Anthropic, etc.)
  - Authentication and authorization systems
