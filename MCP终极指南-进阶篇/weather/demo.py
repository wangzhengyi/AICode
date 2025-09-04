#!/usr/bin/env python3
"""
MCP Weather Server 演示脚本

这个脚本提供了一个简单的交互式演示，展示如何使用MCP Weather服务器。
用户可以通过命令行界面测试不同的功能。

使用方法：
    python demo.py
"""

import asyncio
import json
import subprocess
import sys
from typing import Optional, Dict, Any


class WeatherMCPDemo:
    """MCP Weather服务器演示类"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.initialized = False
    
    async def start_server(self) -> bool:
        """启动MCP服务器"""
        try:
            print("🚀 启动MCP Weather服务器...")
            self.process = subprocess.Popen(
                [sys.executable, "weather.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待服务器启动
            await asyncio.sleep(1)
            print("✅ 服务器启动成功")
            return True
            
        except Exception as e:
            print(f"❌ 服务器启动失败: {e}")
            return False
    
    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求到MCP服务器"""
        try:
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            response_line = self.process.stdout.readline()
            if not response_line:
                raise RuntimeError("服务器无响应")
            
            return json.loads(response_line)
            
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return {"error": {"message": str(e)}}
    
    async def initialize(self) -> bool:
        """初始化MCP连接"""
        print("🔄 初始化MCP连接...")
        
        request = {
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "WeatherDemo", "version": "1.0.0"}
            },
            "jsonrpc": "2.0",
            "id": self.get_next_id()
        }
        
        response = await self.send_request(request)
        
        if "result" in response:
            print("✅ MCP连接初始化成功")
            print(f"服务器信息: {response['result'].get('serverInfo', {})}")
            self.initialized = True
            return True
        else:
            print(f"❌ 初始化失败: {response.get('error', {})}")
            return False
    
    async def list_tools(self) -> bool:
        """列出可用工具"""
        print("🔄 获取可用工具列表...")
        
        request = {
            "method": "tools/list",
            "params": {},
            "jsonrpc": "2.0",
            "id": self.get_next_id()
        }
        
        response = await self.send_request(request)
        
        if "result" in response:
            tools = response["result"].get("tools", [])
            print(f"✅ 找到 {len(tools)} 个可用工具:")
            for i, tool in enumerate(tools, 1):
                print(f"  {i}. {tool['name']}: {tool['description']}")
            return True
        else:
            print(f"❌ 获取工具列表失败: {response.get('error', {})}")
            return False
    
    async def get_weather_forecast(self, latitude: float, longitude: float) -> bool:
        """获取天气预报"""
        print(f"🌤️  获取坐标 ({latitude}, {longitude}) 的天气预报...")
        
        request = {
            "method": "tools/call",
            "params": {
                "name": "get_forecast",
                "arguments": {
                    "latitude": latitude,
                    "longitude": longitude
                }
            },
            "jsonrpc": "2.0",
            "id": self.get_next_id()
        }
        
        response = await self.send_request(request)
        
        if "result" in response:
            content = response["result"].get("content", [])
            if content:
                print("✅ 天气预报获取成功:")
                for item in content:
                    if item.get("type") == "text":
                        # 截取前500个字符避免输出过长
                        text = item.get("text", "")
                        if len(text) > 500:
                            text = text[:500] + "..."
                        print(f"  {text}")
            else:
                print("⚠️  未获取到天气预报数据")
            return True
        else:
            error_msg = response.get("error", {}).get("message", "未知错误")
            print(f"❌ 获取天气预报失败: {error_msg}")
            return False
    
    async def get_weather_alerts(self, state: str) -> bool:
        """获取天气预警"""
        print(f"⚠️  获取 {state} 州的天气预警...")
        
        request = {
            "method": "tools/call",
            "params": {
                "name": "get_alerts",
                "arguments": {
                    "state": state
                }
            },
            "jsonrpc": "2.0",
            "id": self.get_next_id()
        }
        
        response = await self.send_request(request)
        
        if "result" in response:
            content = response["result"].get("content", [])
            if content:
                print("✅ 天气预警获取成功:")
                for item in content:
                    if item.get("type") == "text":
                        # 截取前500个字符避免输出过长
                        text = item.get("text", "")
                        if len(text) > 500:
                            text = text[:500] + "..."
                        print(f"  {text}")
            else:
                print("ℹ️  该州当前没有天气预警")
            return True
        else:
            error_msg = response.get("error", {}).get("message", "未知错误")
            print(f"❌ 获取天气预警失败: {error_msg}")
            return False
    
    def get_next_id(self) -> int:
        """获取下一个请求ID"""
        self.request_id += 1
        return self.request_id
    
    def cleanup(self):
        """清理资源"""
        if self.process:
            self.process.terminate()
            self.process.wait()
    
    async def run_interactive_demo(self):
        """运行交互式演示"""
        print("🌟 MCP Weather Server 交互式演示")
        print("=" * 50)
        
        # 启动服务器
        if not await self.start_server():
            return False
        
        try:
            # 初始化连接
            if not await self.initialize():
                return False
            
            # 列出工具
            await self.list_tools()
            
            print("\n" + "=" * 50)
            print("🎮 开始交互式演示")
            
            while True:
                print("\n请选择要测试的功能:")
                print("1. 获取天气预报")
                print("2. 获取天气预警")
                print("3. 运行预设演示")
                print("4. 退出")
                
                try:
                    choice = input("\n请输入选项 (1-4): ").strip()
                    
                    if choice == "1":
                        # 天气预报
                        print("\n请输入坐标信息:")
                        try:
                            lat = float(input("纬度 (例如: 40.7128): "))
                            lon = float(input("经度 (例如: -74.006): "))
                            await self.get_weather_forecast(lat, lon)
                        except ValueError:
                            print("❌ 无效的坐标格式")
                    
                    elif choice == "2":
                        # 天气预警
                        state = input("\n请输入美国州代码 (例如: CA, NY, TX): ").strip().upper()
                        if len(state) == 2:
                            await self.get_weather_alerts(state)
                        else:
                            print("❌ 无效的州代码格式")
                    
                    elif choice == "3":
                        # 预设演示
                        await self.run_preset_demo()
                    
                    elif choice == "4":
                        print("👋 感谢使用MCP Weather演示！")
                        break
                    
                    else:
                        print("❌ 无效选项，请重新选择")
                
                except KeyboardInterrupt:
                    print("\n\n👋 演示已中断")
                    break
                except EOFError:
                    print("\n\n👋 演示结束")
                    break
            
            return True
            
        except Exception as e:
            print(f"❌ 演示过程中发生错误: {e}")
            return False
        finally:
            self.cleanup()
    
    async def run_preset_demo(self):
        """运行预设演示"""
        print("\n🎬 运行预设演示...")
        print("=" * 30)
        
        # 演示1: 纽约天气预报
        print("\n📍 演示1: 纽约市天气预报")
        await self.get_weather_forecast(40.7128, -74.006)
        
        await asyncio.sleep(2)
        
        # 演示2: 加州天气预警
        print("\n📍 演示2: 加利福尼亚州天气预警")
        await self.get_weather_alerts("CA")
        
        await asyncio.sleep(2)
        
        # 演示3: 洛杉矶天气预报
        print("\n📍 演示3: 洛杉矶天气预报")
        await self.get_weather_forecast(34.0522, -118.2437)
        
        print("\n✅ 预设演示完成！")


def main():
    """主函数"""
    demo = WeatherMCPDemo()
    
    try:
        success = asyncio.run(demo.run_interactive_demo())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 演示已中断")
        demo.cleanup()
        sys.exit(0)


if __name__ == "__main__":
    main()