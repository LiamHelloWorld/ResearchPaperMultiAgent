#Design in an instruction-following manner
#The essence is that to solve a complex task, you can enable two communicative agents collabratively working together step by step to reach solutions. The main concepts include:
# Task: a task can be as simple as an idea, initialized by an inception prompt.
# AI User: the agent who is expected to provide instructions.
# AI Assistant: the agent who is expected to respond with solutions that fulfills the instructions.

# set the task
import os
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from camel.agents import ChatAgent
from camel.toolkits import ArxivToolkit

from camel.societies import RolePlaying
from camel.types import TaskType, ModelType, ModelPlatformType
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory

def setup():
    load_dotenv()

    # 获取环境变量中的 API 密钥
    openai_api_key = os.getenv("OPENAI_API_KEY")

        # Define the model, here in this case we use gpt-4o-mini
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig().as_dict(), # [Optional] the config for model
    )

    task_kwargs = {
        'task_prompt': """Write a research paper about "How and to what extent can multimodality help machine learning models simulate human spatial intelligence in different aspects? You need to link spacial intelligence with congnitive science, the topic could be altered" 

            This is a coursework for OMSCS6795 Cognitive Science.

            Learning Goals:
            The main learning goal of the course is an introduction to the basic concepts, hypotheses, models, methods, issues, and debates in cognitive science. Specific objectives include:
            (1) Introduction to the main information-processing paradigms in cognitive science as well as the main critiques of the paradigms.
            (2) Introduction to the central questions, topics, themes, and perspectives that drive the study of cognitive science, including their historical development and the state of the art.
            (3) Understanding the variety of methodologies used to explore cognitive science, including the capabilities and limitations of different research methods.
            (4) Learning about the relationship between cognitive science and computing, including human-centered computing, design, and educational technology.

            Learning Outcomes:
            By the end of the course, the typical student should know enough about cognitive science to:
            (1) Understand and participate in scholarly conversations on cognitive science.
            (2) Read and understand the cognitive science literature.
            (3) Take advanced courses in cognitive science.
            (4) Take the cognitive science specialization in the Georgia Tech Ph.D. qualifying examination in human-centered computing.
            (5) Analyze and address problems in human-centered computing from a cognitive science perspective.
            (6) Conduct research into cognitive science.
            """,
        'with_task_specify': True,
        'task_specify_agent_kwargs': {'model': model}
    }

    professor = {
    'professor': 'an professor study AI,computer vision and machine learning for over 20 years',
    'user_agent_kwargs': {'model': model}
    }

    researcher = {
    'researcher': 'the best-ever student study computer science curious about the problem and challenges to improve spatial intelligence. your ultimate dream is to realize AGI. By improve and implement spatial intelligence, you can pacitipage in the great step forward of enable AI in real world',
    'assistant_agent_kwargs': {'model': model}
    }

    writer = {
    'writer': 'a professional academic writer to write the disscussion between user_role_name, write the essay in IEEE format',
    'assistant_agent_kwargs': {'model': model}
    }

    society = RolePlaying(
    **task_kwargs,             # The task arguments
    **professor,        # The instruction sender's arguments
    **researcher,   # The instruction receiver's arguments
    **writer,   # The instruction receiver's arguments
    )

    return society

def is_terminated(response):
    """
    Give alerts when the session shuold be terminated.
    """
    if response.terminated:
        role = response.msg.role_type.name
        reason = response.info['termination_reasons']
        print(f'AI {role} terminated due to {reason}')

    return response.terminated

def run(society, round_limit: int=10):

    # Get the initial message from the ai assistant to the ai user
    input_msg = society.init_chat()

    # Starting the interactive session
    for _ in range(round_limit):

        # Get the both responses for this round
        assistant_response, user_response = society.step(input_msg)

        # Check the termination condition
        if is_terminated(assistant_response) or is_terminated(user_response):
            break

        # Get the results
        print(f'[AI User] {user_response.msg.content}.\n')
        # Check if the task is end
        if 'CAMEL_TASK_DONE' in user_response.msg.content:
            break
        print(f'[AI Assistant] {assistant_response.msg.content}.\n')



        # Get the input message for the next round
        input_msg = assistant_response.msg

    return None

run(setup())