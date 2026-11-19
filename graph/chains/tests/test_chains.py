from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content  # take the first document retrieved for grading

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt},
    )
    # assert the binary result is 'yes'
    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    # use retrieval grader with a different question to have the grader return no
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizza", "document": doc_txt},
    )
    # assert the binary result is 'no'
    assert res.binary_score == "no"