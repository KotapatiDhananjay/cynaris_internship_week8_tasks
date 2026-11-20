import pytest

from src.workflow import create_workflow


class FakeResponse:
    content = "Artificial Intelligence enables machines to perform intelligent tasks."


class FakeLLM:
    def invoke(self, prompt):
        return FakeResponse()


@pytest.fixture
def chunks():
    return [
        {
            "source": "ai.txt",
            "content": "Artificial Intelligence enables machines to perform intelligent tasks."
        },
        {
            "source": "ml.txt",
            "content": "Machine Learning learns patterns from data."
        }
    ]


def test_workflow_execution(chunks):

    assistant = create_workflow(
        chunks,
        llm=FakeLLM()
    )

    result = assistant.invoke({
        "question": "What is Artificial Intelligence?"
    })

    assert "answer" in result
    assert len(result["answer"]) > 0


def test_workflow_sources(chunks):

    assistant = create_workflow(
        chunks,
        llm=FakeLLM()
    )

    result = assistant.invoke({
        "question": "What is Artificial Intelligence?"
    })

    assert "ai.txt" in result["sources"]


def test_workflow_empty_question(chunks):

    assistant = create_workflow(
        chunks,
        llm=FakeLLM()
    )

    with pytest.raises(ValueError):
        assistant.invoke({
            "question": ""
        })