import sqlite3
from datetime import datetime


# Create or connect to SQLite database
def init_db():
    conn = sqlite3.connect("feedback.db")  # Creates 'feedback.db' file if it doesn't exist
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            feedback_text TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert feedback into the database
def insert_feedback(username, feedback_text):
    try:
        conn = sqlite3.connect("feedback.db")
        cursor = conn.cursor()

        # Insert feedback into table
        cursor.execute('''
            INSERT INTO feedback (username, feedback_text)
            VALUES (?, ?)
        ''', (username, feedback_text))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database error: {e}")
