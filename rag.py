import numpy as np

from typing import List
from sqlite3 import Connection
from pydantic import BaseModel

from providers.provider import ModelProvider

class RagInput(BaseModel):
    query: str

class KnnResult(BaseModel):
    id: int
    distance: float
    similarity: float

def knn(
    query: list[float],
    db: Connection,
    k: int = 5
) -> List[KnnResult]:
    """
    Perform K-nearest neighbors search using either cosine similarity or euclidean distance.
    
    Args:
        query: The query vector to search with
        db: SQLite database connection
        k: Number of nearest neighbors to return
    
    Returns:
        List of KnnResult objects containing the k nearest neighbors
    """
    # Convert query to numpy array and normalize for cosine similarity
    query_array = np.array(query, dtype=np.float32)
    
    # Convert query to bytes for SQLite
    query_bytes = query_array.tobytes()

    sql = """
        SELECT rowid, distance
        FROM vec_items
        WHERE embedding MATCH :query
        ORDER BY distance
        LIMIT :k
    """
    
    # Execute the query
    cursor = db.execute(sql, {"query": query_bytes, "k": k})
    results = cursor.fetchall()
    
    # Convert results to KnnResult objects
    knn_results = []
    for rowid, distance in results:
        # For cosine similarity, convert distance to similarity (1 - distance)
        similarity = float(distance)
        knn_results.append(KnnResult(
            id=rowid,
            distance=float(distance),
            similarity=similarity
        ))
    
    return knn_results

def rag(input: RagInput, db: Connection, model: ModelProvider) -> str:
    """RAG Steps:
    1. User providers a query they want to search the document database with (input.query)
    2. This query is turned into an embedding (using the model.embed method)
    3. Do KNN search on the database (using the knn method above)
    4. Convert the KNN responses to text by querying database
    5. Construct query with KNN text response
    6. Return response with references to documents that matched
    """
    # Step 2: Generate embedding for the query
    query_embedding = model.embed(input.query)
    
    # Step 3: Perform KNN search
    knn_results = knn(query_embedding, db)
    
    # Step 4: Get the text content for each KNN result
    context_texts = []
    for result in knn_results:
        cursor = db.execute(
            "SELECT content FROM documents WHERE id = ?",
            (result.id,)
        )
        content = cursor.fetchone()
        if content:
            context_texts.append(content[0])
    
    # Step 5: Construct the prompt with context
    context = "\n\n".join(context_texts)
    prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say "I don't know".

Context:
{context}

Question: {input.query}

Answer:"""
    
    # Step 6: Generate and return the response
    response = model.generate(prompt)
    return response