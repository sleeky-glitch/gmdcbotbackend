from pinecone import Pinecone
from typing import Tuple, List, Dict, Any
from app.config import get_settings
from app.services.openai_service import OpenAIService

settings = get_settings()

class PineconeService:
    def __init__(self):
        """
        Initialize Pinecone service with the configured settings.
        """
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            # Connect to the specified index
            self.index = self.pc.Index(settings.index_name)
            # Initialize OpenAI service for embeddings
            self.openai_service = OpenAIService()
        except Exception as e:
            raise Exception(f"Failed to initialize Pinecone service: {str(e)}")

    async def query_documents(self, query: str, namespace: str = None, top_k: int = 10) -> Dict[str, Any]:
        """
        Query documents using the provided query string.

        Args:
            query (str): The query string to search for
            namespace (str, optional): The namespace to search in
            top_k (int): Number of results to return

        Returns:
            Dict containing the query results
        """
        try:
            # Generate embedding vector for the query
            query_vector = await self.openai_service.generate_embedding(query)

            # Perform the query
            query_response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )

            return query_response

        except Exception as e:
            raise Exception(f"Pinecone query failed: {str(e)}")

    def extract_context(self, query_response: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Extract context and references from the query response.

        Args:
            query_response (Dict): The response from Pinecone query

        Returns:
            Tuple containing the combined context string and list of references
        """
        try:
            contexts = []
            references = []

            # Extract text and references from matches
            for match in query_response.matches:
                if hasattr(match, 'metadata'):
                    # Extract text if available
                    if hasattr(match.metadata, 'text'):
                        contexts.append(match.metadata.text)

                    # Extract original filename if available
                    if hasattr(match.metadata, 'original_filename'):
                        references.append(match.metadata.original_filename)

            # Combine all context texts with newlines
            combined_context = "\n".join(contexts) if contexts else ""

            return combined_context, references

        except Exception as e:
            raise Exception(f"Failed to extract context: {str(e)}")

    async def search_and_process_query(self, query: str, namespace: str = None) -> Tuple[str, List[str]]:
        """
        Perform a complete search operation: query documents and extract context.

        Args:
            query (str): The search query
            namespace (str, optional): The namespace to search in

        Returns:
            Tuple containing the context and references
        """
        try:
            # Query the documents
            query_response = await self.query_documents(query, namespace)

            # Extract and return context and references
            return self.extract_context(query_response)

        except Exception as e:
            raise Exception(f"Search and process operation failed: {str(e)}")

    async def upsert_documents(self, vectors: List[Dict[str, Any]], namespace: str = None) -> bool:
        """
        Upsert vectors to the Pinecone index.

        Args:
            vectors (List[Dict]): List of vectors to upsert
            namespace (str, optional): The namespace to upsert to

        Returns:
            bool: True if successful, raises exception otherwise
        """
        try:
            # Perform the upsert operation
            self.index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            return True

        except Exception as e:
            raise Exception(f"Failed to upsert documents: {str(e)}")

    async def delete_documents(self, ids: List[str], namespace: str = None) -> bool:
        """
        Delete vectors from the Pinecone index.

        Args:
            ids (List[str]): List of vector IDs to delete
            namespace (str, optional): The namespace to delete from

        Returns:
            bool: True if successful, raises exception otherwise
        """
        try:
            # Perform the delete operation
            self.index.delete(
                ids=ids,
                namespace=namespace
            )
            return True

        except Exception as e:
            raise Exception(f"Failed to delete documents: {str(e)}")
