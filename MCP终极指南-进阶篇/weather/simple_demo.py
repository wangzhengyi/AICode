#!/usr/bin/env python3
"""
简化的MCP Weather演示脚本

这个脚本不依赖MCP框架，直接演示JSON-RPC通信的概念。
适用于Python 3.7+的环境。

使用方法：
    python simple_demo.py
"""

import json
import sys
from typing import Dict, Any, Optional


class SimpleMCPDemo:
    """简化的MCP演示类"""
    
    def __init__(self):
        self.request_id = 0
        self.tools = [
            {
                "name": "get_forecast",
                "description": "获取指定坐标的天气预报",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number", "description": "纬度"},
                        "longitude": {"type": "number", "description": "经度"}
                    },
                    "required": ["latitude", "longitude"]
                }
            },
            {
                "name": "get_alerts",
                "description": "获取指定州的天气预警",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string", "description": "美国州代码（如CA, NY）"}
                    },
                    "required": ["state"]
                }
            }
        ]
    
    def get_next_id(self) -> int:
        """获取下一个请求ID"""
        self.request_id += 1
        return self.request_id
    
    def create_json_rpc_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """创建JSON-RPC 2.0请求"""
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.get_next_id()
        }
    
    def create_json_rpc_response(self, request_id: int, result: Any = None, error: Any = None) -> Dict[str, Any]:
        """创建JSON-RPC 2.0响应"""
        response = {
            "jsonrpc": "2.0",
            "id": request_id
        }
        
        if error:
            response["error"] = error
        else:
            response["result"] = result
        
        return response
    
    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理初始化请求"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {}
            },
            "serverInfo": {
                "name": "SimpleWeatherMCP",
                "version": "1.0.0"
            }
        }
    
    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理工具列表请求"""
        return {"tools": self.tools}
    
    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理工具调用请求"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "get_forecast":
            return self.mock_get_forecast(arguments)
        elif tool_name == "get_alerts":
            return self.mock_get_alerts(arguments)
        else:
            raise ValueError(f"未知工具: {tool_name}")
    
    def mock_get_forecast(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """模拟天气预报功能"""
        lat = args.get("latitude")
        lon = args.get("longitude")
        
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ValueError("无效的坐标参数")
        
        # 模拟天气预报数据
        forecast_text = f"""📍 坐标: ({lat}, {lon})
🌤️  天气预报 (模拟数据):

今天: 晴转多云，气温 18-25°C，东南风 2-3级
明天: 多云转阴，气温 16-22°C，可能有小雨
后天: 阴转晴，气温 20-28°C，西北风 1-2级

注意: 这是模拟数据，实际使用时会调用真实的天气API。"""
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": forecast_text
                }
            ]
        }
    
    def mock_get_alerts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """模拟天气预警功能"""
        state = args.get("state", "").upper()
        
        if not state or len(state) != 2:
            raise ValueError("无效的州代码")
        
        # 模拟天气预警数据
        if state in ["CA", "FL", "TX"]:
            alert_text = f"""⚠️  {state}州天气预警 (模拟数据):

🌪️  龙卷风警告
- 生效时间: 2024-01-15 14:00 - 18:00
- 影响区域: {state}州中部地区
- 建议: 请待在室内，远离窗户

🌊 洪水警告
- 生效时间: 2024-01-15 20:00 - 次日06:00
- 影响区域: {state}州沿海地区
- 建议: 避免在低洼地区行驶

注意: 这是模拟数据，实际使用时会调用真实的预警API。"""
        else:
            alert_text = f"""✅ {state}州当前无天气预警 (模拟数据)

当前该州没有生效的天气预警信息。
请继续关注天气变化。

注意: 这是模拟数据，实际使用时会调用真实的预警API。"""
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": alert_text
                }
            ]
        }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理JSON-RPC请求"""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "tools/list":
                result = self.handle_tools_list(params)
            elif method == "tools/call":
                result = self.handle_tools_call(params)
            else:
                raise ValueError(f"不支持的方法: {method}")
            
            return self.create_json_rpc_response(request_id, result=result)
            
        except Exception as e:
            error = {
                "code": -32603,
                "message": str(e)
            }
            return self.create_json_rpc_response(request.get("id"), error=error)
    
    def run_demo(self):
        """运行演示"""
        print("🌟 简化的MCP Weather演示")
        print("=" * 50)
        print("这个演示展示了MCP协议的基本JSON-RPC通信流程")
        print("不需要安装MCP框架，适用于Python 3.7+")
        print("=" * 50)
        
        # 演示1: 初始化
        print("\n📋 步骤1: 初始化MCP连接")
        init_request = self.create_json_rpc_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "SimpleDemo", "version": "1.0.0"}
        })
        
        print("发送请求:")
        print(json.dumps(init_request, indent=2, ensure_ascii=False))
        
        init_response = self.process_request(init_request)
        print("\n收到响应:")
        print(json.dumps(init_response, indent=2, ensure_ascii=False))
        
        # 演示2: 获取工具列表
        print("\n" + "=" * 50)
        print("📋 步骤2: 获取可用工具列表")
        tools_request = self.create_json_rpc_request("tools/list", {})
        
        print("发送请求:")
        print(json.dumps(tools_request, indent=2, ensure_ascii=False))
        
        tools_response = self.process_request(tools_request)
        print("\n收到响应:")
        print(json.dumps(tools_response, indent=2, ensure_ascii=False))
        
        # 演示3: 调用天气预报工具
        print("\n" + "=" * 50)
        print("📋 步骤3: 调用天气预报工具")
        forecast_request = self.create_json_rpc_request("tools/call", {
            "name": "get_forecast",
            "arguments": {
                "latitude": 40.7128,
                "longitude": -74.006
            }
        })
        
        print("发送请求:")
        print(json.dumps(forecast_request, indent=2, ensure_ascii=False))
        
        forecast_response = self.process_request(forecast_request)
        print("\n收到响应:")
        print(json.dumps(forecast_response, indent=2, ensure_ascii=False))
        
        # 演示4: 调用天气预警工具
        print("\n" + "=" * 50)
        print("📋 步骤4: 调用天气预警工具")
        alerts_request = self.create_json_rpc_request("tools/call", {
            "name": "get_alerts",
            "arguments": {
                "state": "CA"
            }
        })
        
        print("发送请求:")
        print(json.dumps(alerts_request, indent=2, ensure_ascii=False))
        
        alerts_response = self.process_request(alerts_request)
        print("\n收到响应:")
        print(json.dumps(alerts_response, indent=2, ensure_ascii=False))
        
        # 演示5: 错误处理
        print("\n" + "=" * 50)
        print("📋 步骤5: 错误处理演示")
        error_request = self.create_json_rpc_request("tools/call", {
            "name": "get_forecast",
            "arguments": {
                "latitude": "invalid",  # 无效参数
                "longitude": -74.006
            }
        })
        
        print("发送请求 (包含无效参数):")
        print(json.dumps(error_request, indent=2, ensure_ascii=False))
        
        error_response = self.process_request(error_request)
        print("\n收到错误响应:")
        print(json.dumps(error_response, indent=2, ensure_ascii=False))
        
        print("\n" + "=" * 50)
        print("✅ 演示完成！")
        print("\n📚 关键概念总结:")
        print("1. MCP使用JSON-RPC 2.0协议进行通信")
        print("2. 每个请求都有method、params、jsonrpc和id字段")
        print("3. 响应包含result（成功）或error（失败）字段")
        print("4. 工具调用通过tools/call方法实现")
        print("5. 错误处理遵循JSON-RPC 2.0规范")
        print("\n🚀 要运行真实的MCP服务器，请:")
        print("1. 升级到Python 3.10+")
        print("2. 安装MCP框架: pip install 'mcp[cli]>=1.6.0'")
        print("3. 运行: python weather.py")


def main():
    """主函数"""
    try:
        demo = SimpleMCPDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n👋 演示已中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()