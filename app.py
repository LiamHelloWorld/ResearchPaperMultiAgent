import os
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from camel.agents import ChatAgent
from camel.toolkits import ArxivToolkit


def setup_literature_review_agents():
    load_dotenv()

    # 获取环境变量中的 API 密钥
    openai_api_key = os.getenv("OPENAI_API_KEY")

    sys_msg = 'You are a agent write academic paper literature reviews about certain topics. \n' \

    # Define the model, here in this case we use gpt-4o-mini
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig().as_dict(), # [Optional] the config for model
    )

    # 实例化 ArxivToolkit
    arxiv_toolkit = ArxivToolkit()
    
    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        message_window_size=10, # [Optional] the length for chat memory
        tools = [
            *arxiv_toolkit.get_tools(),  # 使用 get_tools() 方法返回工具列表
        ]
        )

    # Define a user message
    usr_msg = 'Can you define spatial intelligence? and link it with cognitive intelligence?'

    # Sending the message to the agent
    response = agent.step(usr_msg)

    # Check the response (just for illustrative purpose)
    print(response.msgs[0].content)

if __name__ == "__main__":
    setup_literature_review_agents()