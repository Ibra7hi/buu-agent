from langchain_community.vectorstores.pgvector import PGVector
from src.config.settings import DB_CONNECTION_STRING, COLLECTION_NAME


def get_vector_store(embedding_function, collection_name=None):
    """Create and return a PGVector store instance.

    Args:
        embedding_function: The embedding model to use for vector operations.
        collection_name: Override the default collection name from settings.
    """
    vector_store = PGVector(
        connection_string=DB_CONNECTION_STRING,
        embedding_function=embedding_function,
        collection_name=collection_name or COLLECTION_NAME,
        use_jsonb=True
    )

    return vector_store
