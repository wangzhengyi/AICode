# A2A天气Agent演示指南

## 项目概述

这是一个基于A2A协议的天气查询Agent示例，展示了如何构建和运行一个符合A2A规范的智能代理服务。

## 快速开始

### 1. 启动服务

在项目根目录下运行：

```bash
uv run .
```

服务将在 `http://127.0.0.1:10001` 启动。

### 2. 测试Agent身份卡片

A2A协议要求Agent提供身份卡片信息，但本示例中根路径只支持POST方法（符合JSON-RPC规范）：

```bash
# 测试根路径（会返回405 Method Not Allowed，这是正常的）
curl -v http://127.0.0.1:10001/
```

### 3. 测试天气查询功能

使用示例请求文件测试天气查询：

```bash
# 方法1：使用curl
curl -X POST http://127.0.0.1:10001/ \
  -H "Content-Type: application/json" \
  -d @示例/示例请求.json

# 方法2：使用Python脚本
python3 -c "
import json
import requests

with open('示例/示例请求.json', 'r', encoding='utf-8') as f:
    request_data = json.load(f)

response = requests.post('http://127.0.0.1:10001/', json=request_data)
print('Status Code:', response.status_code)
print('Response:')
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
"
```

## 示例请求和响应

### 请求格式（JSON-RPC 2.0）

```json
{
  "id": "90320f73-d6be-462f-bf3d-020ab5043924",
  "jsonrpc": "2.0",
  "method": "agent.execute",
  "params": {
    "contextId": "f0b3fa91-b771-47c5-9ded-cd840e770d1b",
    "taskId": "15dea984-f249-4647-91e4-6a3aacb83369",
    "message": {
      "messageId": "66675023-04aa-4584-8452-746f271f1c9a",
      "role": "user",
      "parts": [
        {
          "kind": "text",
          "text": "西雅图明天的天气怎么样？"
        }
      ]
    }
  }
}
```

### 响应格式

```json
{
  "id": "90320f73-d6be-462f-bf3d-020ab5043924",
  "jsonrpc": "2.0",
  "result": {
    "artifacts": [
      {
        "artifactId": "9bf27e80-c45a-4014-a5ac-dd4b285dc66f",
        "description": "",
        "name": "天气查询结果",
        "parts": [
          {
            "kind": "text",
            "text": "未来 3 天的天气如下：1. 明天（2025年6月1日）：晴天；2. 后天（2025年6月2日）：小雨；3. 大后天（2025年6月3日）：大雨。"
          }
        ]
      }
    ],
    "contextId": "f0b3fa91-b771-47c5-9ded-cd840e770d1b",
    "history": [...],
    "id": "15dea984-f249-4647-91e4-6a3aacb83369",
    "kind": "task",
    "status": {
      "state": "completed"
    }
  }
}
```

## 代码结构说明

### 核心文件

- `__main__.py`: 服务入口点，定义Agent能力和启动服务器
- `agent_executor.py`: Agent执行器，处理具体的天气查询逻辑
- `示例/示例请求.json`: 标准的JSON-RPC请求示例
- `示例/示例返回.json`: 预期的响应格式示例

### A2A协议关键概念

1. **AgentCapabilities**: 定义Agent的基本信息和能力
2. **AgentSkill**: 描述Agent可以执行的技能
3. **AgentCard**: Agent的身份卡片，包含所有元数据
4. **JSON-RPC 2.0**: 通信协议标准
5. **EventQueue**: 异步事件处理机制

## 故障排除

### 常见问题

1. **端口占用错误**
   ```
   OSError: [Errno 48] Address already in use
   ```
   解决方案：修改 `__main__.py` 中的端口号

2. **模块导入错误**
   ```
   ModuleNotFoundError: No module named 'a2a'
   ```
   解决方案：确保使用 `uv run .` 命令启动服务

3. **异步调用错误**
   ```
   Agent execution failed. Error:
   ```
   解决方案：确保在 `agent_executor.py` 中使用 `await event_queue.enqueue_event()`

### 调试技巧

1. 查看服务器日志了解请求处理情况
2. 使用 `curl -v` 查看详细的HTTP交互信息
3. 检查JSON格式是否正确
4. 确认Content-Type头部设置为 `application/json`

## 扩展开发

要扩展这个示例：

1. 修改 `agent_executor.py` 中的 `execute` 方法添加真实的天气API调用
2. 在 `__main__.py` 中更新Agent技能描述
3. 添加更多的输入验证和错误处理
4. 实现更复杂的对话上下文管理

## 总结

这个示例展示了A2A协议的基本实现，包括：
- 符合规范的Agent定义
- JSON-RPC 2.0通信协议
- 异步事件处理
- 结构化的响应格式

通过这个示例，你可以了解如何构建自己的A2A兼容Agent。