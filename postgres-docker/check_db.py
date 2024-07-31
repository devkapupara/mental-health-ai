import psycopg2
from psycopg2 import sql

# Database connection parameters
DB_HOST = "localhost"
DB_NAME = "memory"
DB_USER = "mental"
DB_PASSWORD = "bot"
DB_PORT = "5432"

def verify_postgresql_setup():
    try:
        # Connect to PostgreSQL
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = connection.cursor()

        # Verify the chat_history table exists
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = cursor.fetchall()
        print("Tables in the database:")
        for table in tables:
            print(table[0])

        # Check the structure of the chat_history table
        cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'chat_history';")
        columns = cursor.fetchall()
        print("\nColumns in chat_history table:")
        for column in columns:
            print(f"{column[0]} - {column[1]}")

        # Insert a test record
        cursor.execute(
            "INSERT INTO chat_history (user_input, response) VALUES (%s, %s) RETURNING id;",
            ("Test user input", "Test response")
        )
        record_id = cursor.fetchone()[0]
        connection.commit()
        print(f"\nInserted test record with ID: {record_id}")

        # Query the inserted record
        cursor.execute("SELECT * FROM chat_history WHERE id = %s;", (record_id,))
        record = cursor.fetchone()
        print("\nQueried test record:")
        print(record)

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
    finally:
        if connection:
            cursor.close()
            connection.close()
            print("\nPostgreSQL connection closed.")

if __name__ == "__main__":
    verify_postgresql_setup()