-- Create the chat_history table
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    user_input TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);