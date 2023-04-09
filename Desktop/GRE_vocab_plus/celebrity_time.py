from helpers import getAgentExecutor
from langchain.agents import tool
import requests
from datetime import datetime

@tool
def getTime(input):
    """Get the current time, in 24 hour time, for any timezone, based on a valid timezone input. 
    The input should be the continent name (i.e. Africa, America, Asia, Australia, Europe) followed by a / and the city name.
    For example, (Asia/Dubai, or America/New_York)"""

    try:
        info = datetime.fromisoformat(requests.get(f"http://worldtimeapi.org/api/timezone/{input}").json()["datetime"])
        hour, minute = (info.hour, info.minute)
        if hour > 12:
            return f"{(hour - 12):02}:{minute:02} PM"
        return f"{hour:02}:{minute:02} AM"
    except:
        return "Invalid timezone. Make sure that the input is in the following format: Continent/City"

person_name = input("Who should I impersonate? ")

template = """Answer the following questions as best you can, but speaking as though you are an extremely, extremely exaggerated {person_name}. 
Feel free to exaggerate to an extreme, for comedic intent.
Use quotes from {person_name} if they make sense within the context. Ensure that it is obvious who is speaking. You have access to the following tools:

{{tools}}

Use the following format. Make sure to strictly follow the format, otherwise you will get an error:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]. Make sure that action is one of the provided tool names, otherwise an error will occur.
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, answered like an exaggerated {person_name}. Make sure that you write 'Final Answer:' before you write your answer, or it will not work.

Begin!

Question: {{input}}
{{agent_scratchpad}}""".format(person_name=person_name)
    
tools = [getTime]

agent_executor = getAgentExecutor(tools, template)

print("\n\u001B[1;36m" + agent_executor.run("What's the time in " + input("What timezone would you like to know the time of? ") + "?").strip('"') + "\u001B[0m\n")