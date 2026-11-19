from langsmith import Client

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# pull the rag prompt from langsmith client
# Note: for public prompt, need to set dangerously_pull_public_prompt to True
# Also, you should pin the exact version of the prompt
# Best practice is to paste the prompt directly, which helps avoid incurring costs from pulling the prompt
# and avoid pulling modified versions of the prompt that could be dangerous
prompt = Client().pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)

# create the generation chain
generation_chain = prompt | llm | StrOutputParser()

