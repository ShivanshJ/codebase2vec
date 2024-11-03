# Import necessary libraries
import os
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, PointIdsList, FilterSelector, Filter, Distance,VectorParams, FieldCondition, MatchValue


@dataclass
class VectorNode:
    id: str
    embedding: list
    metadata: dict

    def __init__(self, embedding: list, metadata: dict, id: str = None):
        import uuid
        self.id = id if id is not None else str(uuid.uuid4())
        self.embedding = embedding
        self.metadata = metadata

class VectorStore:
    def __init__(self, collection_name: str, vector_size: int = 128, max_retries: int = 3, retry_delay: int = 2):
        # Initialize Qdrant client
        self.HOST = os.getenv("QDRANT_HOST", "localhost")
        self.PORT = os.getenv("QDRANT_PORT", "6333")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        print (self.HOST, self.PORT, self.QDRANT_API_KEY)

        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = self._connect_with_retry(max_retries, retry_delay)
        self._get_collection(collection_name=self.collection_name, vector_size=self.vector_size)


    def _connect_with_retry(self, max_retries, retry_delay):
        import time
        for attempt in range(max_retries):
            try:
                # Attempt to connect
                client = QdrantClient(self.HOST, port=self.PORT, api_key=self.QDRANT_API_KEY)
                # Test the connection
                client.get_collections()
                print(f"Successfully connected to Qdrant server on attempt {attempt + 1}")
                return client   
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Unable to connect to Qdrant server.")
                    return False
                

    def does_embedding_exist(self, repo_id):
        # Check if embeddings already exist
        search_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="repo_id",
                        match=MatchValue(value=repo_id),
                    ),
                ]
            ),
            limit=1
        )

        if len(search_result[0]) != 0:
            print(f"Embeddings for repo {repo_id} already exist. Skipping.")
            return True
        return False
  
                
    def add_vectors(self, nodes: list[VectorNode], repo_id=1):
        points = []
        for node in nodes:
            points.append(PointStruct(id=node.id, vector=node.embedding, payload=node.metadata))
        self.client.upsert(collection_name=self.collection_name, points=points)


    def search(self, query_vector: list, limit: int = 10):
        # Search for similar vectors in the Qdrant collection
        return self.client.search(collection_name=self.collection_name, query_vector=query_vector, limit=limit)
    
    def get_vectors_by_id(self, point_ids: list):
        # Search for similar vectors in the Qdrant collection
        return self.client.retrieve(collection_name=self.collection_name, ids=point_ids)
    

    def delete_nodes(self, node_ids: list[str]):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=node_ids
            )
        )
    

    # --- Private methods
    def _get_collection(self, collection_name: str, vector_size: int):
        try:
            
            collection_info = self.client.get_collection(collection_name)
            print(f"Connected to existing collection: {collection_name}")

            # Optionally, you can verify the collection parameters here
            if collection_info.config.params.vectors.size != vector_size:
                print(f"Warning: Existing collection has different vector size. Expected: {vector_size}, Actual: {collection_info.config.params.vectors.size}")            
        except Exception as e:
            print ('in here4')
            if not self.client.collection_exists(collection_name=collection_name):
                self._create_collection(collection_name, vector_size)
            else:
                raise RuntimeError(f"Failed to create or recreate collection: {str(e)}")
        
    def _create_collection(self, collection_name: str, vector_size: int):
        try:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create or recreate collection: {str(e)}")
                

    def delete_all_nodes(self):
        """
        Delete all nodes from the collection.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[],
                    must_not=[],
                    should=[]
                )
            )
        )



# ---- Test the vector store
import numpy as np
import unittest
import uuid
class TestVectorStore(unittest.TestCase):
    def setUp(self):
        from dotenv import load_dotenv
        load_dotenv()

        self.vector_size = 768
        self.collection_name = "dev_codebase"
        self.vs = VectorStore(self.collection_name, self.vector_size)

    def test_vector_connection(self):
        # Create test nodes
        id1, id2, id3 = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
        test_nodes = [
            VectorNode(id=id1, embedding=np.random.rand(self.vector_size).tolist(), metadata={"text": "Test document 1"}),
            VectorNode(id=id2, embedding=np.random.rand(self.vector_size).tolist(), metadata={"text": "Test document 2"}),
            VectorNode(id=id3, embedding=np.random.rand(self.vector_size).tolist(), metadata={"text": "Test document 3"}),
        ]
        self.vs.add_vectors(test_nodes)
        
        # Perform a search
        search_vector = np.random.rand(self.vector_size).tolist()
        results = self.vs.search(search_vector, limit=2)

        # Assert results
        self.assertEqual(len(results), 2, "Search should return 2 results")
        for result in results:
            self.assertIsNotNone(result.id, "Result should have an ID")
            self.assertIsNotNone(result.score, "Result should have a score")
            self.assertIsNotNone(result.payload, "Result should have a payload")

    def test_vector_search_by_id(self):
        # Create test nodes
        id1, id2, id3 = 1,2,3
        test_nodes = [
            VectorNode(id=id1, embedding=np.random.rand(self.vector_size).tolist(), metadata={"text": "Test document 1"}),
            VectorNode(id=id2, embedding=np.random.rand(self.vector_size).tolist(), metadata={"text": "Test document 2"}),
            VectorNode(id=id3, embedding=np.random.rand(self.vector_size).tolist(), metadata={"text": "Test document 3"}),
        ]
        self.vs.add_vectors(test_nodes)
        # Search for a specific node by ID
        search_id = [id1]
        res = self.vs.get_vectors_by_id(search_id)
        # Assert results
        self.assertEqual(len(res), 1, "Search should return 1 result")
        self.assertEqual(res[0].id, search_id[0], "Result should have the correct ID")

    def tearDown(self):
        # Clean up the test collection
        self.vs.delete_all_nodes()

if __name__ == "__main__":
    unittest.main()