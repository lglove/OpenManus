from typing import Any, Optional, Callable, Dict, Awaitable

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_saver import FileSaver
from app.tool.python_execute import PythonExecute
from app.tool.web_search import WebSearch
from app.logger import logger

MessageHandler = Callable[[str], Awaitable[None]]
PlanHandler = Callable[[Dict], Awaitable[None]]

class Manus(ToolCallAgent):
    """
    A versatile general-purpose agent that uses planning to solve various tasks.

    This agent extends PlanningAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 2000
    max_steps: int = 20

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), WebSearch(), BrowserUseTool(), FileSaver(), Terminate()
        )
    )

    # 消息处理器
    _message_handler: Optional[MessageHandler] = None
    _plan_handler: Optional[PlanHandler] = None

    def set_message_handler(self, handler: MessageHandler):
        """设置消息处理器"""
        self._message_handler = handler

    def set_plan_handler(self, handler: PlanHandler):
        """设置计划更新处理器"""
        self._plan_handler = handler

    async def think(self) -> bool:
        """思考过程，增加消息处理"""
        # if self._message_handler:
        #     # 记录开始思考的消息
        #     await self._message_handler("Manus's thoughts: 开始分析当前状态和下一步行动...")

        # 调用父类的思考过程
        result = await super().think()

        if result and self._message_handler:
            # 记录思考结果
            if hasattr(self, "messages") and self.messages:
                # 获取最后一条消息的内容
                last_message = self.messages[-1]
                if last_message.content:
                    await self._message_handler(f"Manus's thoughts: {last_message.content}")
            await self._message_handler("Manus's thoughts: 已确定下一步行动计划")

        return result

    async def act(self) -> str:
        """执行动作，增加消息处理"""
        if self._message_handler:
            await self._message_handler("Manus's thoughts: 准备执行计划的下一个步骤...")

        result = await super().act()

        if self._message_handler:
            await self._message_handler(f"Manus's thoughts: 步骤执行完成，结果：{result}")

        # 如果有计划更新，通知计划处理器
        if self._plan_handler and hasattr(self, "active_plan_id"):
            try:
                plan_tool = self.available_tools.get_tool("planning")
                if plan_tool and self.active_plan_id in plan_tool.plans:
                    await self._plan_handler(plan_tool.plans[self.active_plan_id])
            except Exception as e:
                logger.warning(f"Failed to send plan update: {e}")

        return result

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        if not self._is_special_tool(name):
            return
        else:
            if self._message_handler:
                await self._message_handler("Manus's thoughts: 任务完成，正在清理资源...")
            await self.available_tools.get_tool(BrowserUseTool().name).cleanup()
            await super()._handle_special_tool(name, result, **kwargs)
