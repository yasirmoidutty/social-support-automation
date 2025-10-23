from langgraph.graph import StateGraph, END
from nodes import *
import PIL.Image
from IPython.display import Image 
import io


# build graph
graph = StateGraph(AppState)

graph.add_node("data_extractor", data_extractor)
graph.add_node("data_validator", data_validator)
graph.add_node("eligibility_checker", eligibility_checker)
graph.add_node("response_generator", response_generator)
graph.add_node("orchestrator", orchestrator)

graph.add_conditional_edges(
    "orchestrator",
    lambda state: state.get("next")  
)

graph.add_edge("data_extractor", "orchestrator")
graph.add_edge("data_validator", "orchestrator")
graph.add_edge("eligibility_checker", "orchestrator")

graph.add_edge("response_generator", END)

graph.set_entry_point("orchestrator")

# compile graph
evaluater = graph.compile()

# save graph
img = Image(evaluater.get_graph (xray=True). draw_mermaid_png())
pimg = PIL.Image.open(io.BytesIO(img.data))
pimg.save("workflow_graph.png")