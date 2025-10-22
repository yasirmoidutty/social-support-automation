from langgraph.graph import StateGraph, END
from nodes import *


# build graph
graph = StateGraph(AppState)

graph.add_node("data_extractor", data_extractor)
graph.add_node("data_validator", data_validator)
graph.add_node("eligibility_checker", eligibility_checker)
graph.add_node("response_generator", response_generator)
graph.add_node("orchestrator", orchestrator)

graph.add_edge("orchestrator", "data_extractor")
graph.add_edge("data_extractor", "data_validator")
graph.add_edge("data_validator", "eligibility_checker")
graph.add_edge("eligibility_checker", "response_generator")
graph.add_edge("response_generator", END)

graph.set_entry_point("orchestrator")

# compile graph
evaluater = graph.compile()

