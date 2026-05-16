from langchain.agents import AgentState
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing import Callable
from utils.logger_handle import logger
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from utils.prompt_loader import load_system_prompts,load_report_prompts

@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler:Callable[[ToolCallRequest],ToolMessage | Command],
) -> ToolMessage | Command:
    logger.info(f"[monitor_tool]执行工具：{request.tool_call['name']}")
    logger.info(f"[monitor_tool]传入参数：{request.tool_call['args']}")

                                            # 工具执行的监控
    try:
        result = handler(request)

        logger.info(f"[tool monitor]工具{request.tool_call['name']}")
        # 模型调用了fill...工具，改为true就有标记了
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context['report'] = True

        return result

    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        stare:AgentState,     # 整个agent智能体中的状态记录
        runtime:Runtime,      # 记录了整个执行工程中的上下文信息
):         # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(stare['messages'])}条消息")

    logger.debug(f"[log_before_model]{type(stare['messages'][-1]).__name__}|{stare['messages'][-1].content.strip()}")

    return None


@dynamic_prompt                 # 每一次在生成提示词之前调用此函数
def report_prompt_switch(requests:ModelRequest):     # 动态切换提示词
    is_report= requests.runtime.context.get('report',False)
    if is_report:           # 报告生成场景，返回报告提示词内容
        return  load_report_prompts()
    return load_system_prompts()


