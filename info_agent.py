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

# Info retrieval agent message
email_msg = SystemMessage(content="You are a conversational and friendly assistant. " \
"Your only goal is to get the user to provide you their email address. " \
"Once you have received their email address, always retrieve their customer info" \
"Thank them, refer to them by their name and ask them how you can help. " \
"You do not know what you can help them with, just ask them how you can help. " \
"You may only speak in English unless directed otherwise.")

# Define MessagesState, add db_table_names
class State(MessagesState, total=False):

    customer_id: int
    customer_name: str

# Define tools
@tool
def get_customer_info(email: str) -> dict:
    """Look up customer info given their email. ALWAYS make sure you have the email before invoking this.
    
    Args:
        email: str"""
    customer_info_as_string = db.run(f"SELECT * FROM Customer WHERE Email = '{email}';", include_columns = True)
    customer_info = literal_eval(customer_info_as_string[1:-1])
    customer_id = customer_info["CustomerId"]
    customer_name = customer_info["FirstName"]
    return {"customer_id": customer_id, "customer_name": customer_name}



# Bind tools to model
customer_email_model = model.bind_tools([get_customer_info])

# Create Nodes
def customer_email_node(state: State):

    return {"messages": [customer_email_model.invoke([email_msg] + state["messages"])]}

def get_info_node(state: State):
    """Performs the get customer info tool call."""

    # last message contains param tool_calls, a list of dicts
    tool_call = state["messages"][-1].tool_calls[0]
    
    # make tool call, draft tool message
    observation = get_customer_info.invoke(tool_call["args"])

    # interrupt
    decision = interrupt({
        "question": "I've found the following account, is this you?",
        "account_details": observation
    })
    
    if decision == "yes":
        
        tool_message = {"role": "tool", "content" : observation, "tool_call_id": tool_call["id"]}

        # Add the final tool message to our messages
        return {'messages': tool_message, "customer_name": observation['customer_name'], 'customer_id': observation['customer_id']}

    else:

        tool_message = {"role": "tool", "content" : "Please ask the user to try sign in again", "tool_call_id": tool_call["id"]}

        return {'messages': tool_message}

# Define tool conditions
def customer_info_condition(state: State) -> Literal["get_info_node", "__end__"]:
    """Route to email tool handler, or end if no tool is called."""
    
    # Get the last message
    message = state["messages"][-1]
    
    # Check if it's a Done tool call
    if message.tool_calls:
        return "get_info_node"
    else:
        return END
    
# Create graph
builder = StateGraph(State)

# Add nodes
builder.add_node("get_customer_email", customer_email_node)
builder.add_node("get_info_node", get_info_node) 

# Add edges
builder.add_edge(START, "get_customer_email")
builder.add_conditional_edges("get_customer_email", customer_info_condition)
builder.add_edge("get_info_node", "get_customer_email")

# Compile graph, memory is handled by LangGraph
graph = builder.compile()