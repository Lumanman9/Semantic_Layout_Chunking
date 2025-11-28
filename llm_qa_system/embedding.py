from qdrant_client import QdrantClient
from typing import List, Dict, Any, Union, Optional
from dotenv import load_dotenv
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import Distance, VectorParams, PointStruct
import openai
import os
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
import uuid

load_dotenv()


class Embedding:
    def __init__(
            self,
            chunks,
            collection_name,
            embedding_method: str = "openai",
            model: str = "text-embedding-3-small",
            embedding_dim: int = 1536
    ):
        """
        Initialize Embedding with flexible embedding method.

        Args:
            chunks: Text chunks to embed
            collection_name: Qdrant collection name
            embedding_method: Method to generate embeddings ('openai' or 'local')
            model: Embedding model name
            embedding_dim: Dimension of embeddings
        """
        self.chunks = chunks
        self.qdrant = QdrantClient(os.getenv('qdrant_url'), port=os.getenv('qdrant_port'))
        self.collection_name = collection_name
        self.embedding_method = embedding_method

        # Setup embedding generator based on method
        if embedding_method == "openai":
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.generate_embedding = self._generate_openai_embedding
            self.model = model
        elif embedding_method == "local":
            # Default to a good all-MiniLM model if no specific model provided
            model = model if model != "text-embedding-3-small" else "all-MiniLM-L6-v2"
            self.local_model = SentenceTransformer(model)
            self.generate_embedding = self._generate_local_embedding
            # Adjust embedding dim based on local model
            embedding_dim = self.local_model.get_sentence_embedding_dimension()
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_method}")

        self.embedding_dim = embedding_dim
        self._create_collection()

        # Initialize id_strategy
        self.id_strategy = "uuid"  # Default to UUID for safety

    def _create_collection(self) -> None:
        """Create Qdrant collection if it doesn't exist."""
        collections = self.qdrant.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )

    def _generate_openai_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using OpenAI API.

        Args:
            text: Input text

        Returns:
            List of embedding values
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def _generate_local_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using local SentenceTransformer model.

        Args:
            text: Input text

        Returns:
            List of embedding values
        """
        return self.local_model.encode(text).tolist()

    def _detect_id_strategy(self) -> str:
        """
        Detect the ID strategy used in the collection.

        Returns:
            "int" if collection uses integer IDs, "uuid" if it uses UUID strings
        """
        try:
            # Get some points to check their ID types
            results = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=10,
                with_payload=False,
                with_vectors=False
            )

            if not results[0]:
                # Empty collection, can use either strategy
                return self.id_strategy  # Return current default

            # Check ID types in collection
            has_int_ids = any(isinstance(point.id, int) for point in results[0])
            has_str_ids = any(isinstance(point.id, str) for point in results[0])

            if has_int_ids and not has_str_ids:
                return "int"
            elif has_str_ids and not has_int_ids:
                return "uuid"
            else:
                # Mixed ID types - default to UUID for safety
                print("Warning: Mixed ID types detected in collection. Defaulting to UUID strategy.")
                return "uuid"

        except Exception as e:
            print(f"Error detecting ID strategy: {str(e)}. Defaulting to UUID strategy for safety.")
            return "uuid"

    def _get_next_id(self) -> int:
        """
        Get the next available integer ID by scanning through all points.

        Returns:
            Next available ID (max + 1)
        """
        try:
            highest_id = -1
            offset = None

            # Paginate through all points to find the highest ID
            while True:
                # Get a batch of points
                results = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=100,  # Process in batches of 100
                    offset=offset,
                    with_payload=False,
                    with_vectors=False
                )

                points, offset = results

                # If no points returned, we're done
                if not points:
                    break

                # Update highest ID seen so far
                for point in points:
                    if isinstance(point.id, int) and point.id > highest_id:
                        highest_id = point.id

                # If no offset returned, we're done
                if offset is None:
                    break

            # Return highest + 1, or 0 if collection was empty
            return highest_id + 1 if highest_id >= 0 else 0

        except Exception as e:
            print(f"Error getting next ID: {str(e)}. Starting from ID 1000000 for safety.")
            return 1000000  # Start from a high number to avoid potential conflicts

    def store_chunks(
            self,
            chunks: List[str] = None,
            metadata: List[Dict[str, Any]] = None,
            id_strategy: Optional[str] = None
    ) -> None:
        """
        Store text chunks and their embeddings in Qdrant.
        Automatically appends to existing collection if it exists.

        Args:
            chunks: List of text chunks (if None, uses self.chunks)
            metadata: Optional list of metadata dictionaries for each chunk
            id_strategy: ID strategy to use ('auto', 'int', or 'uuid').
                         'auto' detects existing strategy, 'int' uses sequential IDs,
                         'uuid' uses UUID strings
        """
        # Ensure collection exists (create if it doesn't)
        self._create_collection()

        # Use instance chunks if none provided
        if chunks is None:
            chunks = self.chunks

        if metadata is None:
            metadata = [{} for _ in chunks]

        # Determine ID strategy
        if id_strategy is None or id_strategy == 'auto':
            id_strategy = self._detect_id_strategy()
        self.id_strategy = id_strategy

        # Get starting ID if using sequential integers
        start_id = self._get_next_id() if id_strategy == 'int' else 0
        print(f"Using ID strategy: {id_strategy}" +
              (f" (starting at {start_id})" if id_strategy == 'int' else ""))

        points = []

        for i, (chunk, meta) in enumerate(tqdm(zip(chunks, metadata), total=len(chunks))):
            try:
                # Generate embedding
                embedding = self.generate_embedding(chunk)

                # Prepare point data
                point_data = {
                    "text": chunk,
                    "timestamp": datetime.now().isoformat(),
                    **meta
                }

                # Create point with appropriate ID
                if id_strategy == 'uuid':
                    point_id = str(uuid.uuid4())
                else:  # 'int'
                    point_id = start_id + i

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=point_data
                )
                points.append(point)
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                continue

        # Store points in batches
        batch_size = 100
        try:
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            print(f"Successfully stored {len(points)} chunks in collection '{self.collection_name}'")
        except Exception as e:
            print(f"Error storing chunks: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # OpenAI embedding example
    openai_embedding = Embedding(
        chunks=["Example text 1", "Example text 2"],
        collection_name="my_openai_collection",
        embedding_method="openai"
    )
    openai_embedding.store_chunks()  # Auto-detect ID strategy

    # Local embedding with explicit integer IDs
    local_embedding = Embedding(
        chunks=["Example text 1", "Example text 2"],
        collection_name="my_local_collection",
        embedding_method="local",
        model="all-MiniLM-L6-v2"
    )
    local_embedding.store_chunks(id_strategy='int')  # Use sequential integer IDs

    # Explicit UUID strategy
    uuid_embedding = Embedding(
        chunks=["Example text with UUID 1", "Example text with UUID 2"],
        collection_name="my_uuid_collection",
        embedding_method="local"
    )
    uuid_embedding.store_chunks(id_strategy='uuid')  # Use UUID strings