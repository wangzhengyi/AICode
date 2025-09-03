# A2A协议天气Agent实现
# 这是一个基于A2A协议的天气查询Agent服务，展示了如何构建标准化的Agent间通信服务

# 导入A2A协议相关的核心组件
from a2a.server.apps import A2AStarletteApplication  # A2A协议的Starlette应用框架
from a2a.server.request_handlers import DefaultRequestHandler  # 默认请求处理器
from a2a.server.tasks import InMemoryTaskStore  # 内存任务存储器
from a2a.types import (
    AgentCapabilities,  # Agent能力声明
    AgentCard,         # Agent身份卡
    AgentSkill,        # Agent技能定义
)

# 导入自定义的天气Agent执行器
from agent_executor import WeatherAgentExecutor


def main(host: str, port: int):
    """
    天气Agent服务的主函数
    
    Args:
        host: 服务绑定的主机地址
        port: 服务监听的端口号
    """
    
    # 1. 定义Agent能力配置
    # streaming=False 表示该Agent不支持流式响应，所有响应都是一次性返回
    capabilities = AgentCapabilities(streaming=False)
    
    # 2. 定义Agent的具体技能
    # 技能1：天气预告 - 提供未来天气预测功能
    forecast_skill = AgentSkill(
        id='天气预告',                              # 技能唯一标识符
        name='天气预告',                            # 技能显示名称
        description='给出某地的天气预告',              # 技能功能描述
        tags=['天气', '预告'],                      # 技能标签，用于分类和检索
        examples=['给我纽约未来 7 天的天气预告'],      # 使用示例，帮助其他Agent理解如何调用
    )
    
    # 技能2：空气质量报告 - 提供当前空气质量查询功能
    air_quality_skill = AgentSkill(
        id='空气质量报告',                          # 技能唯一标识符
        name='空气质量报告',                        # 技能显示名称
        description='给出某地当前时间的空气质量报告，不做预告',  # 技能功能描述
        tags=['空气', '质量'],                      # 技能标签
        examples=['给我纽约当前的空气质量报告'],        # 使用示例
    )

    # 3. 创建Agent身份卡（AgentCard）
    # AgentCard是Agent的"身份证"，包含了Agent的所有元信息
    agent_card = AgentCard(
        name='天气 Agent',                         # Agent名称
        description='提供天气相关的查询功能',          # Agent功能描述
        url=f'http://{host}:{port}',              # Agent的服务端点URL
        version='1.0.0',                         # Agent版本号，用于版本管理和兼容性控制
        defaultInputModes=['text'],               # 默认支持的输入模式（文本）
        defaultOutputModes=['text'],              # 默认支持的输出模式（文本）
        capabilities=capabilities,                # Agent的能力配置
        skills=[forecast_skill, air_quality_skill],  # Agent提供的技能列表
    )

    # 4. 创建请求处理器
    # DefaultRequestHandler负责处理A2A协议的标准请求流程
    request_handler = DefaultRequestHandler(
        agent_executor=WeatherAgentExecutor(),    # 自定义的业务逻辑执行器
        task_store=InMemoryTaskStore(),          # 任务存储器，用于管理异步任务状态
    )
    
    # 5. 创建A2A服务应用
    # A2AStarletteApplication是基于Starlette的ASGI应用，提供标准的A2A协议接口
    server = A2AStarletteApplication(
        agent_card=agent_card,                   # Agent身份卡
        http_handler=request_handler             # HTTP请求处理器
    )
    
    # 6. 启动ASGI服务器
    import uvicorn
    # 使用uvicorn作为ASGI服务器，启动Agent服务
    # server.build()构建完整的ASGI应用实例
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == '__main__':
    # 启动天气Agent服务
    # 监听本地127.0.0.1:10001，可通过此地址访问Agent服务
    main("127.0.0.1", 10002)
