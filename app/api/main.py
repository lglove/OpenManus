from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncGenerator
import json
import asyncio
import os
from datetime import datetime

from app.agent.manus import Manus
from app.logger import logger

# 设置结果文件保存目录
RESULTS_DIR = "/data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(
    title="OpenManus API",
    description="API for OpenManus - A versatile AI agent framework",
    version="0.1.0"
)

# 添加 CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，在生产环境中应该设置为具体的域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

class ManusRequest(BaseModel):
    prompt: str
    config: Optional[Dict[str, Any]] = None

class ManusResponse(BaseModel):
    result: str
    status: str
    plan: Optional[Dict[str, Any]] = None
    result_file: Optional[str] = None

async def save_execution_result(prompt: str, result: str) -> str:
    """保存执行结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_{timestamp}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write(f"Result:\n{result}")

    return filename

async def stream_response(agent: Manus, prompt: str) -> AsyncGenerator[str, None]:
    """生成流式响应"""
    try:
        # 创建一个队列来接收消息
        message_queue = asyncio.Queue()
        all_messages = []  # 用于收集所有消息

        # 定义消息处理器
        async def message_handler(message: str):
            # 检查消息类型并格式化
            if "thoughts:" in message.lower():
                msg_obj = {
                    "type": "thought",
                    "content": message
                }
            elif "tools being prepared:" in message.lower() or "tools selected" in message.lower():
                msg_obj = {
                    "type": "tool_usage",
                    "content": message
                }
            else:
                msg_obj = {
                    "type": "message",
                    "content": message
                }
            all_messages.append(msg_obj)  # 保存消息
            await message_queue.put(msg_obj)

        # 定义计划更新处理器
        async def plan_update_handler(plan: Dict):
            msg_obj = {
                "type": "plan_update",
                "content": plan
            }
            all_messages.append(msg_obj)  # 保存消息
            await message_queue.put(msg_obj)

        # 设置消息处理器
        agent.set_message_handler(message_handler)
        agent.set_plan_handler(plan_update_handler)

        # 设置父类（ToolCallAgent）的消息处理器
        if hasattr(agent, "_message_handler"):
            agent._message_handler = message_handler

        # 启动 agent 执行
        task = asyncio.create_task(agent.run(prompt))

        # 持续从队列获取消息并发送
        while True:
            try:
                # 等待消息，但设置超时以检查任务是否完成
                message = await asyncio.wait_for(message_queue.get(), timeout=0.1)
                yield f"data: {json.dumps(message)}\n\n"
            except asyncio.TimeoutError:
                # 检查任务是否完成
                if task.done():
                    # 获取最终结果
                    try:
                        final_result = task.result()
                        # 保存执行结果
                        result_file = await save_execution_result(prompt, final_result)
                        final_message = {
                            "type": "final",
                            "content": final_result,
                            "result_file": result_file
                        }
                        yield f"data: {json.dumps(final_message)}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                    break
                continue
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                break
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

@app.post("/api/v1/manus/stream")
async def stream_request(request: ManusRequest):
    try:
        # 创建 Manus 实例
        agent = Manus()

        # 如果有配置，应用配置
        if request.config:
            for key, value in request.config.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)

        # 返回流式响应
        return StreamingResponse(
            stream_response(agent, request.prompt),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/manus", response_model=ManusResponse)
async def process_request(request: ManusRequest):
    try:
        # 创建 Manus 实例
        agent = Manus()

        # 如果有配置，应用配置
        if request.config:
            for key, value in request.config.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)

        # 运行 agent
        result = await agent.run(request.prompt)

        # 保存执行结果
        result_file = await save_execution_result(request.prompt, result)

        # 获取计划状态（如果有）
        plan = None
        if hasattr(agent, "active_plan_id"):
            try:
                plan_tool = agent.available_tools.get_tool("planning")
                if plan_tool and agent.active_plan_id in plan_tool.plans:
                    plan = plan_tool.plans[agent.active_plan_id]
            except Exception as e:
                logger.warning(f"Failed to get plan: {e}")

        return ManusResponse(
            result=result,
            status="success",
            plan=plan,
            result_file=result_file
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy"}
