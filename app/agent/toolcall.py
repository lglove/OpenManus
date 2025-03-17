import json
from typing import Any, List, Optional, Union, Callable, Awaitable

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    # 添加消息处理器
    _message_handler: Optional[Callable[[str], Awaitable[None]]] = None

    def set_message_handler(self, handler: Callable[[str], Awaitable[None]]):
        """设置消息处理器"""
        self._message_handler = handler

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        # 检测用户输入的语言并设置语言上下文
        user_messages = [msg for msg in self.messages if msg.role == "user"]
        is_chinese = False
        if user_messages:
            last_user_msg = user_messages[-1].content
            is_chinese = any(ord(c) > 127 for c in last_user_msg)
            # 添加语言要求到系统提示
            language_prompt = (
                "你必须始终使用中文回复。这包括：\n"
                "1. 所有的思考过程\n"
                "2. 工具使用的描述\n"
                "3. 错误信息\n"
                "4. 最终结果\n"
                "即使是在调用工具时，也要用中文描述你的思考过程。" if is_chinese
                else "You must always respond in English. This includes:\n"
                "1. All thought processes\n"
                "2. Tool usage descriptions\n"
                "3. Error messages\n"
                "4. Final results\n"
                "Even when calling tools, describe your thought process in English."
            )
            # 将语言要求放在系统提示的最前面，确保模型优先考虑语言要求
            system_prompt = f"{language_prompt}\n\n{self.system_prompt}"
        else:
            system_prompt = self.system_prompt

        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=[Message.system_message(system_prompt)]
                if system_prompt
                else None,
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                error_msg = (
                    f"🚨 令牌限制错误：{token_limit_error}" if is_chinese
                    else f"🚨 Token limit error: {token_limit_error}"
                )
                logger.error(error_msg)
                if self._message_handler:
                    thought_prefix = "思考过程：" if is_chinese else "thoughts:"
                    await self._message_handler(f"{thought_prefix} {error_msg}")
                self.memory.add_message(
                    Message.assistant_message(
                        error_msg
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = response.tool_calls

        # Log response info and send through message handler
        if response.content:
            # 将思考过程作为独立的消息发送
            thoughts = response.content.strip()
            logger.info(f"✨ {self.name}'s thoughts: {thoughts}")
            if self._message_handler:
                # 分段发送思考内容，每段都标记为思考过程
                thought_prefix = "思考过程：" if is_chinese else "thoughts:"
                for thought in thoughts.split('\n'):
                    if thought.strip():
                        await self._message_handler(f"{thought_prefix} {thought.strip()}")

        if response.tool_calls:
            tool_msg = (
                f"🛠️ 已选择 {len(response.tool_calls)} 个工具" if is_chinese
                else f"🛠️ {len(response.tool_calls)} tools selected"
            )
            logger.info(tool_msg)
            if self._message_handler:
                await self._message_handler(tool_msg)

            tools_msg = (
                f"正在准备以下工具：{[call.function.name for call in response.tool_calls]}" if is_chinese
                else f"Tools being prepared: {[call.function.name for call in response.tool_calls]}"
            )
            logger.info(tools_msg)
            if self._message_handler:
                await self._message_handler(tools_msg)

        try:
            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if response.tool_calls:
                    warning_msg = (
                        "🤔 工具当前不可用" if is_chinese
                        else "🤔 Tool usage attempted when not available"
                    )
                    logger.warning(warning_msg)
                    if self._message_handler:
                        await self._message_handler(warning_msg)
                if response.content:
                    self.memory.add_message(Message.assistant_message(response.content))
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(
                    content=response.content, tool_calls=self.tool_calls
                )
                if self.tool_calls
                else Message.assistant_message(response.content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(response.content)

            return bool(self.tool_calls)
        except Exception as e:
            error_msg = (
                f"🚨 思考过程出错：{e}" if is_chinese
                else f"🚨 Error in thinking process: {e}"
            )
            logger.error(error_msg)
            if self._message_handler:
                await self._message_handler(f"thoughts: {error_msg}")
            self.memory.add_message(
                Message.assistant_message(
                    error_msg
                )
            )
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        # 检查用户语言
        user_messages = [msg for msg in self.messages if msg.role == "user"]
        is_chinese = False
        if user_messages:
            last_user_msg = user_messages[-1].content
            is_chinese = any(ord(c) > 127 for c in last_user_msg)

        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or (
                "没有内容或命令需要执行" if is_chinese
                else "No content or commands to execute"
            )

        results = []
        for command in self.tool_calls:
            if self._message_handler:
                tool_msg = (
                    f"🔧 正在激活工具：'{command.function.name}'..." if is_chinese
                    else f"🔧 Activating tool: '{command.function.name}'..."
                )
                await self._message_handler(tool_msg)

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            completion_msg = (
                f"🎯 工具 '{command.function.name}' 已完成任务！结果：{result}" if is_chinese
                else f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )
            logger.info(completion_msg)
            if self._message_handler:
                await self._message_handler(completion_msg)

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result, tool_call_id=command.id, name=command.function.name
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        # 检查用户语言
        user_messages = [msg for msg in self.messages if msg.role == "user"]
        is_chinese = False
        if user_messages:
            last_user_msg = user_messages[-1].content
            is_chinese = any(ord(c) > 127 for c in last_user_msg)

        if not command or not command.function or not command.function.name:
            return "错误：无效的命令格式" if is_chinese else "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"错误：未知工具 '{name}'" if is_chinese else f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")

            # Execute the tool
            logger.info(f"🔧 正在激活工具：'{name}'..." if is_chinese else f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # Format result for display
            observation = (
                f"执行命令 `{name}` 的输出结果：\n{str(result)}" if result and is_chinese
                else f"命令 `{name}` 执行完成，无输出" if not result and is_chinese
                else f"Observed output of cmd `{name}` executed:\n{str(result)}" if result
                else f"Cmd `{name}` completed with no output"
            )

            # Handle special tools like `finish`
            await self._handle_special_tool(name=name, result=result)

            return observation
        except json.JSONDecodeError:
            error_msg = (
                f"解析 {name} 的参数时出错：无效的 JSON 格式" if is_chinese
                else f"Error parsing arguments for {name}: Invalid JSON format"
            )
            logger.error(
                f"📝 参数格式错误！'{name}' 的参数不是有效的 JSON 格式，参数：{command.function.arguments}" if is_chinese
                else f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"错误：{error_msg}" if is_chinese else f"Error: {error_msg}"
        except Exception as e:
            error_msg = (
                f"⚠️ 工具 '{name}' 遇到问题：{str(e)}" if is_chinese
                else f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            )
            logger.error(error_msg)
            return f"错误：{error_msg}" if is_chinese else f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]
