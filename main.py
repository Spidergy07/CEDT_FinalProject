from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Simple RAG API",
    description="A simple API for RAG (Retrieval-Augmented Generation) operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    context: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str] = []

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = {}

class DocumentResponse(BaseModel):
    id: str
    status: str
    message: str

# In-memory storage (replace with actual database in production)
documents = {}
document_counter = 0

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {"message": "Simple RAG API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "rag-api"}

@app.post("/documents", response_model=DocumentResponse)
async def add_document(document: DocumentRequest):
    """Add a document to the knowledge base"""
    global document_counter
    document_counter += 1
    doc_id = f"doc_{document_counter}"
    
    documents[doc_id] = {
        "content": document.content,
        "metadata": document.metadata
    }
    
    return DocumentResponse(
        id=doc_id,
        status="success",
        message="Document added successfully"
    )

@app.get("/documents")
async def list_documents():
    """List all documents in the knowledge base"""
    return {
        "total_documents": len(documents),
        "documents": [
            {"id": doc_id, "metadata": doc_data["metadata"]}
            for doc_id, doc_data in documents.items()
        ]
    }

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document by ID"""
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": doc_id,
        "content": documents[doc_id]["content"],
        "metadata": documents[doc_id]["metadata"]
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the knowledge base"""
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    del documents[doc_id]
    return {"message": f"Document {doc_id} deleted successfully"}

@app.post("/query", response_model=QueryResponse)
async def query_documents(query: QueryRequest):
    """Query the knowledge base with a question"""
    if not documents:
        return QueryResponse(
            answer="No documents available in the knowledge base.",
            confidence=0.0,
            sources=[]
        )
    
    # Simple keyword matching (replace with actual RAG implementation)
    question_words = set(query.question.lower().split())
    best_match = None
    best_score = 0.0
    matched_docs = []
    
    for doc_id, doc_data in documents.items():
        content_words = set(doc_data["content"].lower().split())
        score = len(question_words.intersection(content_words)) / len(question_words)
        
        if score > 0:
            matched_docs.append((doc_id, score))
        
        if score > best_score:
            best_score = score
            best_match = doc_data["content"]
    
    if best_match:
        # Simple answer generation (replace with actual LLM integration)
        answer = f"Based on the documents, here's what I found: {best_match[:200]}..."
        sources = [doc_id for doc_id, _ in sorted(matched_docs, key=lambda x: x[1], reverse=True)[:3]]
    else:
        answer = "I couldn't find relevant information in the knowledge base."
        sources = []
    
    return QueryResponse(
        answer=answer,
        confidence=best_score,
        sources=sources
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
