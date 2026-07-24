"""Clear the semantic cache without touching the main document index."""

import sys
import psycopg2
from src.config.settings import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

def clean_semantic_cache():
    print(f"Connecting to PostgreSQL on port {DB_PORT}...")
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True
        cursor = conn.cursor()

        print("Clearing 'semantic_cache' collection from pgvector...")
        
        # 1. Delete all embedded queries/responses associated with the cache
        cursor.execute("""
            DELETE FROM langchain_pg_embedding 
            WHERE collection_id IN (
                SELECT uuid FROM langchain_pg_collection WHERE name = 'semantic_cache'
            );
        """)
        
        # 2. Delete the cache collection metadata
        cursor.execute("DELETE FROM langchain_pg_collection WHERE name = 'semantic_cache';")
        
        print("\n✅ Success: Semantic cache cleared! Your main document index is safe.")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error while clearing cache: {e}")
        sys.exit(1)

if __name__ == "__main__":
    clean_semantic_cache()
