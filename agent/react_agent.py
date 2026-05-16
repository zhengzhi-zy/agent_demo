from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize,get_weather,fill_context_for_report,generate_external_data,
                                     get_user_id,get_user_location,fetch_external_data,get_current_month)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch
class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize,get_weather,fill_context_for_report,get_user_id,get_user_location,fetch_external_data,get_current_month,generate_external_data],
            middleware=[monitor_tool,log_before_model,report_prompt_switch]
        )

    def execute_stream(self,query:str):
        input_dict={
            "messages":[
                {
                    "role":"user","content":query
                }
            ]
        }
        # context是上下文runtime的信息，是做提示词切换的标记
        for chunk in self.agent.stream(input_dict,stream_mode="values",context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("生成个人使用报告"):
        print(chunk,end="",flush=True)