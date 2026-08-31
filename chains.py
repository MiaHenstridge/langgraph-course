import datetime
from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion


# initialize the LLM
llm = ChatOpenAI(model="gpt-5.4-mini")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])   # parse response into the AnswerQuestion schema defined

# main prompt template for actor agent
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect an critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.
"""
        ),
        MessagesPlaceholder(variable_name="messages"),  # history messages for reviser agent
        ("system", "Answer the user's question above using the required format"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),   # fill in some input variables (i.e., time)
)


# plug in first instruction to the prompt template
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)


# first responder chain to be piped to the llm
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"    # force the response to the AnswerQuestion pydantic object
)


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about Graph Neural Networks,"
        " List their application domains"
    )

    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)

