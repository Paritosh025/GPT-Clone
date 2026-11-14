from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import json
import uuid
from pathlib import Path

# we need fast api and uvicorn for the web server for the agent to run and stream responses to the client, helps in setting up a web server and 
# handling HTTP requests and responses
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# we need langchain core for the essential components of the agent, such as messages, tools, and agents to facilitate the agent's functionality 
# and interactions
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

# here we import the get_agent function from the agent module, which is responsible for creating and configuring the agent instance that will 
# handle user queries and interactions
from agent import get_agent
from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations)

from rag import add_document_to_rag
from tools import set_current_thread_id

app = FastAPI()

templates = Jinja2Templates(directory="templates") # main work of this is to set up the templates directory for rendering HTML templates in the FastAPI application. It allows the application to serve dynamic web pages by using Jinja2 templating engine, enabling the integration of Python code with HTML to create interactive and data-driven web pages.

# it is used to create the necessary directories for storing uploaded files and data, ensuring that the application has the required folder 
# structure to function correctly. If the directories already exist, it will not raise an error due to exist_ok=True.
Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# initializes the database by calling the init_db function, which sets up the necessary database tables and connections for storing and 
# retrieving chat messages, conversations, and other relevant data used by the agent. This is crucial for maintaining the state and history 
# of interactions with the agent.
init_db()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

# this to list all the conversations that have been created and stored in the database, providing an endpoint for clients to retrieve a 
# list of conversations along with their metadata such as thread_id, title, created_at, and updated_at. This allows users to view and manage
# their past interactions with the agent.
@app.get("/conversations")
async def conversations():
    items = list_conversations()

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat()
            }
            for item in items
        ]
    }


# this endpoint retrieves the chat history for a specific conversation identified by the thread_id. It fetches the messages associated with
# the given thread_id from the database and returns them in a structured format, allowing clients to access the conversation history for display or further processing.
@app.get("/history/{thread_id}")
async def history(thread_id: str):
    messages = get_chat_history(thread_id)

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }


# this endpoint handles the uploading of documents to the server. It accepts a file and a thread_id as input, checks if the file type is supported, saves the file to the uploads directory, and adds it to the RAG (Retrieval-Augmented Generation) system for processing. It also updates the conversation associated with the thread_id to indicate that a document has been uploaded. The response includes information about the uploaded file and the number of chunks created for RAG processing.
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    try:
        allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]

        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV."
                },
                status_code=400
            )

        file_id = str(uuid.uuid4())
        safe_filename = filename.replace(" ", "_")
        file_path = f"uploads/{file_id}_{safe_filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        create_or_update_conversation(thread_id, "Uploaded document")

        result = add_document_to_rag(
            file_path=file_path,
            thread_id=thread_id
        )

        return JSONResponse({
            "success": True,
            "message": f"Uploaded {result['filename']} and created {result['chunks']} chunks."
        })

    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "message": str(e)
            },
            status_code=500
        )


# this function formats the payload for Server-Sent Events (SSE) by converting the given dictionary payload into a JSON string and prepending it with "data: " and appending two newline characters. This format is required for SSE to properly send data to clients in real-time, allowing for streaming updates from the server to the client.
def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# this function determines whether a given message chunk should be streamed to the client based on its type and metadata. It filters out chunks that are related to tools, tool calls, or invalid tool calls, ensuring that only normal AI-generated text chunks are sent to the frontend. This helps maintain a clean and relevant stream of messages for the user.
def should_stream_chunk(chunk, metadata) -> bool:
    """
    This prevents raw tool/search/RAG JSON from appearing in the frontend.

    We only stream normal AI text chunks.
    We do NOT stream:
    - ToolMessage
    - messages from tool nodes
    - tool call chunks
    - raw tool outputs
    """

    metadata = metadata or {}

    node_name = str(metadata.get("langgraph_node", "")).lower()

    if "tool" in node_name:
        return False

    if isinstance(chunk, ToolMessage):
        return False

    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False

    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}

    if additional_kwargs.get("tool_calls"):
        return False

    return True


def extract_text_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", "")

    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

        return "".join(text_parts)

    return ""



# this endpoint handles streaming chat responses from the agent. It receives a user message, thread_id, and selected model, validates the input, and then uses the agent to generate a response in a streaming manner. The response is sent back to the client as Server-Sent Events (SSE), allowing for real-time updates as the agent generates its reply. The conversation is also saved in the database for future reference.
@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body."},
            status_code=400
        )

    user_message = data.get("message", "")
    thread_id = data.get("thread_id", "default")
    selected_model = data.get("model", "gpt-4o-mini")

    if not user_message.strip():
        return JSONResponse(
            {"error": "Message is required."},
            status_code=400
        )

    agent = get_agent(selected_model)

    create_or_update_conversation(thread_id, user_message)
    save_chat_message(thread_id, "user", user_message)

    set_current_thread_id(thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }


# this generator function is responsible for streaming the agent's response to the client. It processes the user's message, streams the response in chunks, and yields each chunk as a Server-Sent Event (SSE) to the client. It also handles saving the final answer to the database and manages any exceptions that may occur during the streaming process.

    def event_generator():
        final_answer = ""

        try:
            inputs = {
                "messages": [
                    HumanMessage(content=user_message)
                ]
            }

            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                stream_mode="messages"
            ):
                if not should_stream_chunk(chunk, metadata):
                    continue

                token = extract_text_from_chunk(chunk)

                if token:
                    final_answer += token
                    yield sse_data({"token": token})

            if final_answer.strip():
                save_chat_message(thread_id, "assistant", final_answer)

            yield sse_data({"done": True})

        except Exception as e:
            yield sse_data({"error": str(e)})
            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# this block checks if the script is being run directly (as opposed to being imported as a module) and starts the FastAPI application using Uvicorn. It specifies the host, port, and enables auto-reload for development purposes, allowing the server to automatically restart when code changes are detected.
if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )