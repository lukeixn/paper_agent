from typing import TypedDict
class chief_state(TypedDict):
    user_input:str
    route:str
    result:str
cheif_bus: chief_state = {
    "user_input": "hello",
    "route": "",
    "result": ""
}
def cheif_node(cheif_bus):

    text=cheif_bus["user_input"]
    if text=="hello":
        cheif_bus["route"]="hello_route"
        cheif_bus["result"]="Hello, how can I assist you today?"    
        cheif_bus["route"]="reply_agent"
    return cheif_bus
cheif_node(cheif_bus)
print(cheif_bus)
