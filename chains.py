from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# reflection prompt to crtique the linkedin post
reflection_prompt = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            "You are a viral twitter LinkedIn influencer grading a post. Generate critique and recommendations for the user's LinkedIn post."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),      # placeholder for history messages
    ]    
)

# generation prompt to revise the linkedin post
generation_prompt = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            "You are a LinkedIn Data Science/AI influencer assistant tasked with writing excellent LinkedIn posts."
            " Generate the best LinkedIn post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",

        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# initialize an LLM
llm = ChatOpenAI(model="gpt-5.4-mini")
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

