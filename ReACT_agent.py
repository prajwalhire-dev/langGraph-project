## Objectivies
# - Learn how to create Tools with LangGraph
# - How to create a ReAct Graph
# - work with different types of Messages such as ToolMessages
# - Test out robustness of our graph

## Main Goal : create a robust ReAct agent with tools
from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # BaseMessage is the parent class for all message types // fundamental class
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as content and too_call_id
from langchain_core.messages import SystemMessage #message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages #reducer function
from langgraph.prebuilt import ToolNode

load_dotenv()

#Annotated : provides additional context without affecting the type itself
#Sequence : To automatically handle the state updates for sequence such as by adding new messages to a chat history.

#About Reducer function
#1. Rule that controls how updates from nodes are combined with the existing state.
#2. Tells us how to merge new data into the current state.
#3. Without a reducer, updates would have replaced the existing value entirely!

# --- EXPLANATION: Annotated, Sequence, BaseMessage, add_messages ---
# 1. Sequence[BaseMessage]:
#    - Sequence means a list (or any ordered collection).
#    - BaseMessage is the parent class for all message types (user, system, tool, etc.).
#    - Example:
#        messages = [SystemMessage("Hello"), ToolMessage("Result", tool_call_id="123")]
#    - This lets you store the entire chat history as a list of messages.
#
# 2. Annotated[..., add_messages]:
#    - Annotated lets you attach extra metadata to a type.
#    - Here, we attach the 'add_messages' reducer function.
#    - This tells LangGraph how to update the 'messages' list:
#        - Instead of replacing the whole list, new messages are appended.
#        - Ensures chat history grows as the agent interacts.
#    - Example:
#        If current messages = [A, B] and new = [C], after update: [A, B, C]
#
# 3. BaseMessage:
#    - The fundamental message type in LangChain.
#    - All messages (user, system, tool) inherit from BaseMessage.
#    - This allows the agent to handle different message types in a unified way.
#
# 4. What happens if you DON'T use these?
#    - Without Sequence: You couldn't store multiple messages (no chat history).
#    - Without BaseMessage: You couldn't mix different message types (user, tool, system).
#    - Without Annotated + add_messages: Each update would overwrite the message list, losing history.
#      Example: If you set messages = [C], you lose [A, B].
#    - With this setup, LangGraph automatically appends new messages, keeping the full interaction history.
#
# --- SUMMARY ---
# This pattern makes state updates robust and automatic.
# You don't have to manually merge message histories.
# Keeps all agent interactions in order for context.
# ---------------------------------------------------


class AgentState(TypedDict):
    # messages:
    # - Type: Annotated[Sequence[BaseMessage], add_messages]
    #   - Sequence[BaseMessage]:
    #     - This means 'messages' is a list (or sequence) of BaseMessage objects.
    #     - BaseMessage is the parent class for all message types (user, system, tool, etc.).
    #     - Example: [SystemMessage("Hello"), ToolMessage("Result", tool_call_id="123")]
    #   - Annotated[..., add_messages]:
    #     - Annotated lets us attach extra metadata to the type.
    #     - Here, we attach the 'add_messages' reducer function.
    #     - This tells LangGraph how to update the 'messages' list:
    #         - Instead of replacing the whole list, new messages are appended.
    #         - Ensures chat history grows as the agent interacts.
    #     - Example: If current messages = [A, B] and new = [C], after update: [A, B, C]
    #   - Why?
    #     - This pattern makes state updates robust and automatic.
    #     - You don't have to manually merge message histories.
    #     - Keeps all agent interactions in order for context.
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int):
    """This is an addition function"""
    return a+b

@tool
def subtract(a:int, b:int):
    """Subtraction function"""
    return a-b

@tool
def multiply(a:int, b:int):
    """Multiplication function"""
    return a*b

tools = [add, subtract, multiply]

model = ChatOpenAI(model='gpt-4o').bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"]) #state["messages"] so the query (human message) will be added to the model
    return {"messages":[response]} #here add_message, reduer functions handles the appending, so no need tor anything.

def should_continue(state:AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges( #conditional edge is just a one way direction from agent to tool node
    "our_agent",
    should_continue,
    {
        "continue":"tools", #it will go for tools call
        "end":END,
    }
)

graph.add_edge("tools", "our_agent")

# MODEL DIAGRAM
# START -> Agent
#        / /    \
#      TOOLS     END
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user","Add 3+4, Add 16+9, Subtract 100-10, and Multiply 500*300")]}
print_stream(app.stream(inputs, stream_mode = "values"))

