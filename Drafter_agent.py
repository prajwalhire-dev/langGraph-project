#we are not going for ReACT agent 
#since the ReACT agent would return to agent again after the tool call.
#But for our requirement we don't need tool to agent call again.
#We will end at Tool call.

from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

#This is the global variable to store document content
document_content = ""
"""
In LangGraph (and agent frameworks generally), a global variable is 
sometimes used to store data that needs to persist outside the agent’s 
message history—for example, a document’s content, a database connection, or a 
cache. This is useful when you want to keep track of information that isn’t 
just part of the chat (like a file being built up or modified).

Ingestive state refers to the agent’s ability to “ingest” (take in) 
and store external information during its workflow—such as uploading, editing, or 
summarizing a document. The global variable acts as a simple form of ingestive 
state: it holds the content that the agent or tools can read from or write to as 
the workflow progresses.

Connection:

1.The global variable is the storage for the ingestive state.
2.Tools or agent nodes can update or read this variable to keep track of the 
document or other external data.
3.Without it, the agent would only have access to its message history, not 
persistent or shared data.

Example:
If your agent’s job is to draft a document step-by-step, each tool call might 
update document_content. This way, the agent can “ingest” new information and 
build up the document over time.
"""

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content : str) -> str: #content will be given by LLM
    """Updates the document with the provided content"""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is : \n{document_content}"

@tool
def save(filename:str) -> str: #file name will be given by LLM
    """
    Save the current docuement to a text file and finish the process.

    Args:
        Filename : Name for the text file.
    """
    global document_content
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as f:
            f.write(document_content)
        print(f"\n Document has been saved to : {filename}")
        return f"Document has been saved successfully to '{filename}'"
    except Exception as e:
        return f"Error saving document : {str(e)}"
    

tools = [update, save]

model = ChatOpenAI(model='gpt-4o').bind_tools(tools)

def our_agent(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are a drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
""")
    #if nothing in the state['messages'], we will start with updating the document
    if not state['messages']:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document?")
        print(f"\n USER : {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_prompt] + list(state['messages']) + [user_message]

    response = model.invoke(all_messages)
    print(f"\n AI : {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": list(state['messages']) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conservation"""

    messages = state['messages']

    if not messages:
        return "continue"
    
    #Thsi looks for the most recent tool message ....
    for message in reversed(messages):
        # .... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and 
            "document" in message.content.lower()):
            return "end" #goes to the end edge which leads to the endpoint
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format."""
    if not messages:
        return 
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n Tool Result : {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agents", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agents")
graph.add_edge("agents", "tools")

graph.add_conditional_edges(
    "tools", 
    should_continue,
    {
        "continue":"agents",
        "end":END,
    }
)

app = graph.compile()

def run_document_agent():
    print("\n ======= DRAFTER ========")
    state = {"messages":[]}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n ===== DRAFTER FINISHED =======")

if __name__ == "__main__":
    run_document_agent()