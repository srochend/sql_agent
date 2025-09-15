from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from ast import literal_eval
import sqlite3
import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Literal
from langgraph.types import interrupt

# Create db
def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script) # high-level wrapper around our connection, this will now be the central object which represents the db
    
    return create_engine(
        "sqlite://", # tells sqlalchemy "im working with sqlite", it's empty as we're going to supply our own connection
        creator=lambda: connection, # this tells sqlalchemy to use, and re-use our connection. this is important as the db lives in ram so the connection must stay the same
        poolclass=StaticPool, # similarly we do not want a dynamic connection pool to be created
        connect_args={"check_same_thread": False}, # same as before, allow use of connections across threads
    )

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

# Define model
model = ChatOpenAI(temperature=0, streaming=True, model="gpt-5-nano")

# SQL Agent message
sql_msg = SystemMessage(content="You are a conversational and friendly assistant for a physical-copy music store. " \
"You are equipped to help the customer do the following things, upon request:" \
" - look up information about their account and past purchases" \
" - browse our store's music catalogue"
" - suggest music and build playlists for the customer" \
" - facilitate transactions on behalf of the customer" \
"If the customer asks you to do something outside this scope try to redirect them towards the topic of music" \
"Please use the customers name and only speak English unless otherwise prompted")

# Define MessagesState, add db_table_names
class State(MessagesState, total=False):

    customer_id: int
    customer_name: str
    summary: str

# Define tools
@tool
def make_sql_query(sql_query: str) -> dict:
    """Query the SQL database to retrieve information relevant to you or the customer.
    Before querying a table make sure to get_table_info.
    Some tables will contain a key for customer id.
    The customer id is defined in the state.
    You may not make a sql query for information that contains any other customer id.
    
    Args:
        sql_query: str"""
    sql_query_output = db.run(f"{sql_query}", include_columns = True)
    return {"sql_query_output": sql_query_output}

@tool
def get_table_info(table_name: str) -> dict:
    """Run this call on a table within the database before running a sql query on that table.
    The table name HAS to be in the list db_table_names to be a valid table.

    Args:
        table_name: str"""
    
    try:
        table_info = db.get_table_info([table_name])
    except Exception as e:
        print(e)
    
    return {"table_name": table_name, "table_info": table_info}

# Collect all tools
sql_tools = [make_sql_query, get_table_info]
sql_tools_by_name = {tool.name: tool for tool in sql_tools}

# Bind tools to model
sql_model = model.bind_tools(sql_tools)

# Create Nodes
def sql_model_node(state: State):

    # Get table names
    db_table_names = db.get_usable_table_names()

    # Inject state into the system prompt
    state_context = f"""
The current customer's ID is {state["customer_id"]}.
The current customer's name is {state['customer_name']}.
The valid database tables are: {db_table_names}.
Always use the customer's ID from state when making queries.
Do not ask the user for their ID.
"""
    contextual_sys_msg = SystemMessage(content=sql_msg.content + "\n\n" + state_context)


    return {"messages": [sql_model.invoke([contextual_sys_msg] + state["messages"])]}

def summarizer_node(state: State):
    
    # First get the summary if it exists
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # If a summary already exists, add it to the prompt
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        # If no summary exists, just create a new one
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages and add our summary to the state 
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
    return {"summary": response.content, "messages": delete_messages}

sql_tools_node = ToolNode(sql_tools)

def sql_agent_condition(state: State) -> Literal["sql_tools", "summarizer_node"]:
    """Route to sql tool handler, or conversation summarizer if no tool is called."""
    
    # Get the last message
    message = state["messages"][-1]
    
    # Check if it's a Done tool call
    if message.tool_calls:
        return "sql_tools"
    else:
        return "summarizer_node"

# Create graph
builder = StateGraph(State)

# Add nodes
builder.add_node("sql_model_node", sql_model_node)
builder.add_node("sql_tools", sql_tools_node)
builder.add_node("summarizer_node", summarizer_node)

# Add edges
builder.add_edge(START, "sql_model_node")
builder.add_conditional_edges("sql_model_node", sql_agent_condition)
builder.add_edge("sql_tools", "sql_model_node")
builder.add_edge("summarizer_node", END)

# Compile graph, memory is handled by LangGraph
graph = builder.compile()
