"""Reset the database — wipe all embeddings and conversation history."""

import sys
import psycopg2
from src.config.settings import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

def reset_database():
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

        print("Clearing pgvector embedding tables...")
        # Check if tables exist and truncate them
        cursor.execute("""
            DO $$ 
            BEGIN
                IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'langchain_pg_embedding') THEN
                    TRUNCATE TABLE langchain_pg_embedding CASCADE;
                END IF;
                IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'langchain_pg_collection') THEN
                    TRUNCATE TABLE langchain_pg_collection CASCADE;
                END IF;
            END $$;
        """)
        
        print("Clearing conversation checkpoint tables...")
        # Drop checkpoint tables so the checkpointer starts completely fresh next time
        checkpoint_tables = ["checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"]
        for table in checkpoint_tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
            
        print("\n✅ Success: Database reset complete (embeddings and conversation history cleared).")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error while resetting database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    reset_database()
