"""
Decides whether the generated answer is grounded in retrieved documents
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in generation answer
    """
    binary_score: bool = Field(
        description="Answer is grounded in the facts: True if grounded, False otherwise"
    )


# create grader with structured output
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# grader prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# chat prompt template
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)

# create the chain
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader