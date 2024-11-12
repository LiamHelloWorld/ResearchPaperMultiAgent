import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import AnthropicConfig
from camel.agents import EmbodiedAgent
from camel.generators import SystemMessageGenerator as sys_msg_gen
from camel.messages import BaseMessage as bm
from camel.types import RoleType
import pandas as pd

def load_data(file_path):
    """Load the input data file into a pandas DataFrame."""
    return pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

def save_data(df, output_file):
    """Save the processed DataFrame to the specified output file."""
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False)
    elif output_file.endswith('.xlsx'):
        df.to_excel(output_file, index=False)

def set_agent():
    #claude_api_key = ""

        # Set the role name and the task
    role = 'Programmer'
    task = 'Writing and executing codes.'
    

    model = ModelFactory.create(
        openai_api_key = os.getenv("OPENAI_API_KEY"),
        model_platform=ModelPlatformType.ANTHROPIC,
        model_type=ModelType.CLAUDE_3_OPUS,
        model_config_dict={"temperature": 0.4, "max_tokens": 4096}, # [Optional] the config for model
        )

    # Create the meta_dict and the role_tuple
    meta_dict = dict(role=role, task=task)
    role_tuple = (role, RoleType.EMBODIMENT)
    # Generate the system message based on this
    sys_msg = sys_msg_gen().from_dict(meta_dict=meta_dict, role_tuple=role_tuple)
    embodied_agent = EmbodiedAgent(model = model,
                                system_message=sys_msg,
                               tool_agents=None,
                               code_interpreter=None,
                               verbose=True)
    
    usr_msg = bm.make_user_message(
        role_name='user',
        content=('1. load data using  defined load_data function'
                '2. find column with head "Item No.", rename into "Manufacturer Product" for output dataframe'
                '3. find columns with head "Qty JAN", "Qty FEB" etc.., sum the value and put into "Qty" column for output dataframe'
                '4. use defined save_data function to export output data frame as "output.xlsx"')
    )
    response = embodied_agent.step(usr_msg)
    print(response.msg.content)

set_agent()
  
