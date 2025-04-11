from fastapi import FastAPI
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain import hub
from langchain_chroma import Chroma
from pydantic import BaseModel
from typing import TypedDict, List


class Question(BaseModel):
    question: str


class State(TypedDict):
    question: str
    answer: str
    context: List[Document]


class QAAgent:
    def __init__(self, llm: AzureChatOpenAI, embedding: AzureOpenAIEmbeddings, chroma_db_path: str):
        self.llm = llm
        self.embedding = embedding
        self.chroma = Chroma(collection_name="test-task-collection", embedding_function=self.embedding, persist_directory=chroma_db_path)
        self.prompt = hub.pull("rlm/rag-prompt")
    

    def call_model(self, state: State, config: RunnableConfig):
        question = state.get("question")
        context = self.chroma.similarity_search(question, k=5)
        messages = self.prompt.invoke({"question": question, "context": context})
        response = self.llm.invoke(messages, config)

        return {"question": question, "answer": response.content, "context": context}


    def build_graph(self, checkpointer: MemorySaver = None):
        builder = StateGraph(State)

        # Define nodes
        builder.add_node("assistant", self.call_model)

        # Define edges
        builder.add_edge(START, "assistant")
        builder.add_edge("assistant", END)

        graph = builder.compile(checkpointer=checkpointer)

        return graph
    

app = FastAPI()


@app.get("/")
def read_root():
    return {"Message": "Welcome to test-task!"}

@app.post("/ask/")
async def ask(question: Question):
    # Initialize the LLM and embedding model
    llm = AzureChatOpenAI(
        api_version="2024-10-21",
        azure_deployment="gpt-4o"
    )
    embedding = AzureOpenAIEmbeddings(
        api_version="2024-10-21",
        azure_deployment="text-embedding-3-small-1"
    )
    chroma_db_path = "./chroma_db"
    agent = QAAgent(llm, embedding, chroma_db_path)
    # Initialize the graph
    graph = agent.build_graph()
    response = graph.invoke(
        {
            "question": question.question
        },
        config={"configurable": {"thread_id": "1"}}
    )

    return response
