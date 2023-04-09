from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, tool
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
import os

os.environ["OPENAI_API_KEY"] = "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"

class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format_messages(self, **kwargs):
        intermediate_steps = kwargs.pop("intermediate_steps")

        if (intermediate_steps):
            print("\n\u001B[36m" + intermediate_steps[-1][0].log.split("\n")[0].replace("Thought: ", "") + "\n\u001B[0m")

        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

def getAgentExecutor(tools, template):
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )

    class CustomOutputParser(AgentOutputParser):

        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)|Action: (None)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            if (match.group(1)):
                action = match.group(1).strip()
                action_input = match.group(2)
            else:
                action = "None"
                action_input = "None"

            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

    output_parser = CustomOutputParser()

    llm = ChatOpenAI(temperature=0.7)

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )

    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False)