import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import tools

Path("data").mkdir(exist_ok=True) # to create a data directory data folder if it doesn't exist

# Use OpenAI by default.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ALLOWED_MODELS = {
    "gpt-4o-mini",                 # OpenAI's GPT-4o model
    "gpt-4o",
}


SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named GPT-Clone similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, current prices, current people, current versions, new releases, or anything time-sensitive, use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""

# Main work of this function is to validate the model name and ensure it is in the allowed list. If not, it falls back to the default model.
def normalize_model_name(model_name: str | None) -> str:
    """
    Validate selected model from frontend.
    If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


# This function builds a LangGraph agent for the selected OpenAI model. It initializes ChatOpenAI with the selected model,
# binds tools to it, and sets up the workflow with chatbot and tool nodes. The workflow is then compiled with a SQLite checkpointer for state 
# persistence.

def build_agent(model_name: str):
    """
    Build one LangGraph agent for a selected OpenAI model.
    """

    selected_model = normalize_model_name(model_name)

    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        model=selected_model,
        temperature=0.3, # Adjusted temperature for more deterministic responses, close to 0 means more deterministic, while higher values introduce more randomness.
        streaming=True # Enable streaming for real-time response generation
    )

    llm_with_tools = llm.bind_tools(tools) # Bind the tools to the LLM so that it can use them when needed. eg calculator/ stock prize weather

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = llm_with_tools.invoke(messages) # Invoke the LLM with the current messages to generate a response.

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState) # Initialize a StateGraph with MessagesState to manage the conversation state. 

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite", # persistence memory for the agent, so that it can remember previous conversations and states.
        check_same_thread=False # sqlite local database and this flag allows multiple threads to access the database connection safely.
    )

    checkpointer = SqliteSaver(conn) # connection object is passed to SqliteSaver to handle saving and loading of the workflow state to the SQLite database.

    return workflow.compile(checkpointer=checkpointer) # Compile the workflow with the checkpointer to enable state persistence across sessions. 
                                                       # This allows the agent to remember previous interactions and maintain context.


_AGENT_CACHE = {} # This dictionary caches the built agents for different models. When an agent is requested, it checks if it already exists in the cache; 
                  # if not, it builds a new one and stores it in the cache for future use.

def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    selected_model = normalize_model_name(model_name) # get correct model name based on the input or default to the predefined model. This ensures that the agent is always built with a valid model name.

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]