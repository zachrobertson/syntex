import requests
import json
from providers.openai import OpenAIModel

# Initialize the model with the same configuration as the server
model = OpenAIModel(
    text_model="gpt-4o",
    embedding_model="text-embedding-3-small"
)

# Test documents
test_documents = [
    {
        "id": "python_basics",
        "content": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."
    },
    {
        "id": "python_lists",
        "content": "In Python, lists are ordered collections of items that can be of different types. They are mutable, meaning you can change their content. Lists are created using square brackets [] and items are separated by commas."
    },
    {
        "id": "python_functions",
        "content": "Functions in Python are defined using the def keyword. They can take parameters and return values. Python functions are first-class objects, meaning they can be passed as arguments to other functions."
    },
    {
        "id": "python_classes",
        "content": "Classes in Python are used to create objects. They encapsulate data and functions that operate on that data. Classes are defined using the class keyword and can inherit from other classes."
    }
]

def add_document(doc):
    """Add a document to the database"""
    response = requests.post(
        "http://localhost:8080/add-document",
        json=doc
    )
    print(f"Added document {doc['id']}: {response.json()}")

def test_search(query_text):
    """Test the search functionality with a query"""
    # Generate embedding for the query
    query_embedding = model.embed(query_text)
    
    # Create search request
    search_request = {
        "query_embedding": query_embedding,
        "top_k": 2
    }
    
    # Send search request
    response = requests.post(
        "http://localhost:8080/search",
        json=search_request
    )
    
    print("\nSearch results:")
    print(json.dumps(response.json(), indent=2))

def main():
    # Add test documents
    print("Adding test documents...")
    for doc in test_documents:
        add_document(doc)
    
    # Test search queries
    print("\nTesting search functionality...")
    test_queries = [
        "What are Python lists?",
        "How do functions work in Python?",
        "Tell me about Python classes"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        test_search(query)

if __name__ == "__main__":
    main() 