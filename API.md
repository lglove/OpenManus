# OpenManus API 文档

OpenManus API 提供了一个简单的 HTTP 接口，允许你通过 RESTful API 调用 OpenManus 的功能。

## 基本信息

- 基础 URL: `http://localhost:8000`
- API 版本: v1
- 内容类型: `application/json` 或 `text/event-stream`（用于流式输出）

## 认证

目前版本不需要认证。如果需要，可以在配置中添加认证机制。

## 端点

### 1. 流式处理请求

使用 Server-Sent Events (SSE) 提供实时的处理进度和结果。

**端点:** `/api/v1/manus/stream`

**方法:** POST

**请求体:**

```json
{
  "prompt": "string",
  "config": {
    "max_steps": 20,
    "max_observe": 2000
    // 其他可选配置
  }
}
```

**响应:**
Server-Sent Events 流，每个事件的格式为：

```json
{
  "type": "message|plan_update|final|error",
  "content": "string|object"
}
```

事件类型说明：

- `message`: 处理过程中的状态消息
- `plan_update`: 计划更新信息
- `final`: 最终处理结果
- `error`: 错误信息

**示例请求:**

```javascript
// 使用 JavaScript 发起请求
const eventSource = new EventSource("/api/v1/manus/stream");

eventSource.onmessage = function (event) {
  const data = JSON.parse(event.data);
  switch (data.type) {
    case "message":
      console.log("状态消息:", data.content);
      break;
    case "plan_update":
      console.log("计划更新:", data.content);
      break;
    case "final":
      console.log("最终结果:", data.content);
      eventSource.close();
      break;
    case "error":
      console.error("错误:", data.content);
      eventSource.close();
      break;
  }
};
```

```python
# 使用 Python requests 发起请求
import requests

response = requests.post(
    "http://localhost:8000/api/v1/manus/stream",
    json={
        "prompt": "帮我写一个简单的Python程序",
        "config": {
            "max_steps": 10
        }
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        # 处理 SSE 数据
        data = line.decode('utf-8').replace('data: ', '')
        print(data)
```

### 2. 标准处理请求

处理用户的请求并返回最终结果。

**端点:** `/api/v1/manus`

**方法:** POST

**请求体:**

```json
{
  "prompt": "string",
  "config": {
    "max_steps": 20,
    "max_observe": 2000
    // 其他可选配置
  }
}
```

**响应:**

```json
{
  "result": "string",
  "status": "string",
  "plan": {
    "plan_id": "string",
    "title": "string",
    "steps": ["string"],
    "step_statuses": ["string"],
    "step_notes": ["string"]
  }
}
```

**示例请求:**

```bash
curl -X POST "http://localhost:8000/api/v1/manus" \
     -H "Content-Type: application/json" \
     -d '{
         "prompt": "帮我写一个简单的Python程序",
         "config": {
             "max_steps": 10
         }
     }'
```

### 3. 健康检查

检查 API 服务的健康状态。

**端点:** `/api/v1/health`

**方法:** GET

**响应:**

```json
{
  "status": "healthy"
}
```

**示例请求:**

```bash
curl "http://localhost:8000/api/v1/health"
```

## 错误处理

API 使用标准的 HTTP 状态码表示请求的结果：

- 200: 成功
- 400: 请求错误
- 500: 服务器错误

错误响应格式：

```json
{
  "detail": "错误描述"
}
```

## 部署说明

1. 使用 Docker 部署：

```bash
# 构建镜像
docker build -t openmanus-api .

# 运行容器
docker run -p 8000:8000 openmanus-api
```

2. 直接运行：

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python api_server.py
```

## 配置选项

可以通过 `config` 参数配置以下选项：

- `max_steps`: 最大执行步骤数（默认：20）
- `max_observe`: 最大观察长度（默认：2000）
- 其他 Manus 类支持的配置项

## 注意事项

1. API 服务是异步的，某些长时间运行的任务可能需要较长时间才能完成
2. 建议在生产环境中添加适当的认证机制
3. 可以根据需要调整配置参数来优化性能
4. 流式输出接口使用 SSE 技术，确保客户端支持 EventSource
5. 对于长时间运行的任务，建议使用流式输出接口以获得实时反馈
