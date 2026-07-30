from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    """The state of our agent."""
    messages: Annotated[list[BaseMessage], add_messages]

def build_custom_agent(model, tools, system_prompt: str, checkpointer=None):
    """
    Builds a custom ReAct agent using StateGraph.
    This replaces the prebuilt `create_react_agent` and gives you full control
    to add more complex routing, self-reflection nodes, or specialized logic.
    """
    # 1. Bind tools to the model
    model_with_tools = model.bind_tools(tools)
    
    # 2. Define the tool node
    tool_node = ToolNode(tools)
    
    # 3. Define the main model calling node
    def call_model(state: AgentState):
        messages = state["messages"]
        # Prepend system prompt if it's not already there
        if system_prompt and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
            
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
        
    # 4. Define the routing logic
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM decided to call a tool, route to the tools node
        if getattr(last_message, "tool_calls", None):
            return "tools"
            
        # Otherwise, we're done
        return "__end__"
        
    # 5. Build the graph
    workflow = StateGraph(AgentState)
    
    # Add our nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Add our edges
    workflow.add_edge(START, "agent")
    
    # Conditional routing after the agent speaks
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    
    # After tools run, always return to the agent
    workflow.add_edge("tools", "agent")
    
    # 6. Compile into a runnable application
    app = workflow.compile(checkpointer=checkpointer)
    return app
