## Objects
# - Use different message types - HumanMessage, and AIMessage
# - Maintain a full conversation history using both message types
# - Use GPT-4o model using LangChain's ChatOpenAI
# - Create a sophisticated conversation loop

## Main Goal : create a form of memory for the chatbot

from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

def process(state: AgentState) -> AgentState:
    """
    Process the messages in the state and return the updated state.
    """
    response = llm.invoke(state['messages'])
    print(f"\nAI: {response.content}")
    # Append AI response to the messages
    state['messages'].append(AIMessage(content=response.content))
    print("CURRENT STATE:", state['messages'])
    return state

graph = StateGraph(AgentState)
graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)
agent = graph.compile() 

conversation_history = [] # this will hold the conversation history // memory
user_input = input("Enter : ")
while user_input.lower() != 'exit':
    # Append user input to the conversation history
    conversation_history.append(HumanMessage(content=user_input))
    
    # Invoke the agent with the current conversation history
    result = agent.invoke({'messages': conversation_history})

    # Get the AI response from the last message in the history
    conversation_history = result['messages'] 

    
    user_input = input("Enter : ")

with open("conversation_history.txt", "w") as f:
    f.write("Your conversation history:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("\nEnd of conversation history.\n")

print("Conversation history saved to conversation_history.txt")
