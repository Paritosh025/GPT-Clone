from datetime import datetime
from pathlib import Path

# sqlalchemy imports for database operations, it is used to define the database schema and interact with the SQLite database.
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

Path("data").mkdir(exist_ok=True)

DATABASE_URL = "sqlite:///data/chatbot_memory.db" # creating a local serverless SQLite database file named chatbot_memory.db in the data directory. 
                                        # This database will be used to store conversations, chat messages, and long-term memory for the chatbot.

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
) # Here, we create a SQLAlchemy engine that connects to the SQLite database. The connect_args parameter is set to allow multiple threads to access the database connection safely, which is important for web applications that may handle concurrent requests.

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Conversation(Base): # database schema for the conversations table, which stores information about each conversation thread, including its unique ID, title, and timestamps for creation and last update.
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, unique=True, index=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class LongTermMemory(Base):
    __tablename__ = "long_term_memory"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    memory = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db(): # initializes the database by creating the necessary tables based on the defined schema. It uses SQLAlchemy's Base.metadata.create_all method to create the tables in the SQLite database if they do not already exist.
    Base.metadata.create_all(bind=engine)

def create_or_update_conversation(thread_id: str, first_message: str | None = None):
    db = SessionLocal()

    try: # updates the conversation's title and timestamps if it already exists, or creates a new conversation entry if it doesn't. It uses SQLAlchemy's session management to handle database transactions safely.
        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if not conversation:
            title = "New Chat"

            if first_message:
                title = first_message.strip()[:40] # create left sidebar title for the conversation, if the first message is longer than 40 characters, it truncates it and adds an ellipsis to indicate that there is more content. This helps in providing a concise and readable title for the conversation in the user interface.
                if len(first_message.strip()) > 40:
                    title += "..."

            conversation = Conversation(
                thread_id=thread_id,
                title=title,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            db.add(conversation)

        else:
            conversation.updated_at = datetime.utcnow()

        db.commit()

    finally:
        db.close()


def list_conversations(): # listing all the conversations in the database, ordered by their last updated timestamp in descending order. This allows users to see their most recent conversations first when they access the chat application.
    db = SessionLocal()

    try:
        return (
            db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    finally:
        db.close()

# if user want to save any message will use this function to save the message in the database, it takes the thread ID, role (user or assistant), and content of the message as parameters. It creates a new ChatMessage entry in the database and updates the corresponding conversation's updated_at timestamp to reflect the latest activity.
def save_chat_message(thread_id: str, role: str, content: str):
    db = SessionLocal()


    try:
        msg = ChatMessage(
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=datetime.utcnow()
        )

        db.add(msg)

        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )

        if conversation:
            conversation.updated_at = datetime.utcnow()

        db.commit()

    finally:
        db.close()


# to get the chat history for a specific conversation thread, it queries the ChatMessage table for all messages associated with the given thread ID, ordered by their creation timestamp in ascending order. This allows users to view the entire conversation history in chronological order.
def get_chat_history(thread_id: str):
    db = SessionLocal()

    try:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

    finally:
        db.close()


# functions to save and retrieve long-term memory for the chatbot. The save_memory function stores a memory entry associated with a specific conversation thread, while the search_memory function retrieves the most recent memory entries for a given thread ID, allowing the chatbot to recall important information from previous interactions.
def save_memory(thread_id: str, memory: str):
    db = SessionLocal()

    try:
        item = LongTermMemory(
            thread_id=thread_id,
            memory=memory,
            created_at=datetime.utcnow()
        )

        db.add(item)
        db.commit()

        return "Memory saved successfully."

    finally:
        db.close()


# search_memory function retrieves the most recent memory entries for a given thread ID, allowing the chatbot to recall important information from previous interactions.
def search_memory(thread_id: str, query: str):
    db = SessionLocal()

    try:
        memories = (
            db.query(LongTermMemory)
            .filter(LongTermMemory.thread_id == thread_id)
            .order_by(LongTermMemory.created_at.desc())
            .limit(20)
            .all()
        )

        if not memories:
            return "No saved memory found."

        return "\n".join([f"- {m.memory}" for m in memories])

    finally:
        db.close()