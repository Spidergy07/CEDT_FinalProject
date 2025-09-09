#!/usr/bin/env python3
"""
Simple test script for the RAG API
Run this after starting the API server to test its functionality.
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoints"""
    print("🔍 Testing health check endpoints...")
    
    # Test root endpoint
    response = requests.get(f"{BASE_URL}/")
    print(f"GET / : {response.status_code} - {response.json()}")
    
    # Test health endpoint
    response = requests.get(f"{BASE_URL}/health")
    print(f"GET /health : {response.status_code} - {response.json()}")

def test_document_operations():
    """Test document CRUD operations"""
    print("\n📚 Testing document operations...")
    
    # Add documents
    documents = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            "metadata": {"topic": "programming", "language": "python"}
        },
        {
            "content": "FastAPI is a modern web framework for building APIs with Python. It's built on Starlette and Pydantic, providing automatic API documentation.",
            "metadata": {"topic": "web framework", "language": "python"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn and make decisions from data.",
            "metadata": {"topic": "machine learning", "field": "AI"}
        }
    ]
    
    doc_ids = []
    for i, doc in enumerate(documents):
        response = requests.post(f"{BASE_URL}/documents", json=doc)
        if response.status_code == 200:
            doc_id = response.json()["id"]
            doc_ids.append(doc_id)
            print(f"✅ Added document {i+1}: {doc_id}")
        else:
            print(f"❌ Failed to add document {i+1}: {response.status_code}")
    
    # List all documents
    response = requests.get(f"{BASE_URL}/documents")
    if response.status_code == 200:
        docs = response.json()
        print(f"📋 Total documents: {docs['total_documents']}")
    
    # Get a specific document
    if doc_ids:
        response = requests.get(f"{BASE_URL}/documents/{doc_ids[0]}")
        if response.status_code == 200:
            print(f"📄 Retrieved document: {doc_ids[0]}")
    
    return doc_ids

def test_query_operations(doc_ids):
    """Test query operations"""
    print("\n❓ Testing query operations...")
    
    queries = [
        "What is Python?",
        "Tell me about FastAPI",
        "What is machine learning?",
        "How do you build APIs?",
        "What is artificial intelligence?"
    ]
    
    for query in queries:
        response = requests.post(f"{BASE_URL}/query", json={"question": query})
        if response.status_code == 200:
            result = response.json()
            print(f"\nQ: {query}")
            print(f"A: {result['answer'][:100]}...")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Sources: {result['sources']}")
        else:
            print(f"❌ Failed to query: {query}")

def test_cleanup(doc_ids):
    """Clean up test documents"""
    print("\n🧹 Cleaning up...")
    
    for doc_id in doc_ids:
        response = requests.delete(f"{BASE_URL}/documents/{doc_id}")
        if response.status_code == 200:
            print(f"🗑️ Deleted document: {doc_id}")

def main():
    """Run all tests"""
    print("🚀 Starting RAG API Tests")
    print("=" * 50)
    
    try:
        # Test health endpoints
        test_health_check()
        
        # Test document operations
        doc_ids = test_document_operations()
        
        # Test query operations
        test_query_operations(doc_ids)
        
        # Optional: clean up documents
        cleanup = input("\nDo you want to clean up test documents? (y/N): ")
        if cleanup.lower() == 'y':
            test_cleanup(doc_ids)
        
        print("\n✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the server is running at http://localhost:8000")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
