from fastapi import FastAPI
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain import hub
from langchain_chroma import Chroma
from pydantic import BaseModel
from typing import List


class Question(BaseModel):
    question: str


class State(MessagesState):
    question: str
    answer: str
    context: List[Document]


@tool
def search_vanna_knowledge_base(question: str):
    """Search the Vanna knowledge base for relevant information. Vanna AI is a SQL Generator project. Knowledge base contains information about the project, tools, and source code.
    This function uses the Chroma database to perform a similarity search based on the question provided.
    Args:
        question (str): The question to search for.
    Returns:
        List[Document]: A list of relevant documents from the knowledge base.
    """
    # Perform a similarity search in the Chroma database
    embedding = AzureOpenAIEmbeddings(
        api_version="2024-10-21",
        azure_deployment="text-embedding-3-small-1"
    )
    chroma_db_path = "./chroma_db"
    chroma = Chroma(collection_name="test-task-collection", embedding_function=embedding, persist_directory=chroma_db_path)
    context = chroma.similarity_search(question, k=5)

    return context


class QAAgent:
    '''
    A class to handle the question-answering agent using Azure OpenAI and Vanna knowledge base.
    '''
    def __init__(self, llm: AzureChatOpenAI):
        '''
        Initialize the QAAgent with an LLM and tools.
        Args:
            llm (AzureChatOpenAI): The language model to use for generating responses.
        '''
        self.llm = llm
        self.tools = [search_vanna_knowledge_base]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.prompt = hub.pull("rlm/rag-prompt")
        self.sys_msg = open("./prompts/system_message.txt", "r").read()
    
    
    def call_model(self, state: State):
        '''
        Call the language model with the provided state.
        Args:
            state (State): The current state of the agent.
        Returns:
            dict: The response from the language model.
        '''
        messages = [SystemMessage(content=self.sys_msg)] + state.get("messages")
        response = self.llm_with_tools.invoke(messages)

        return {"messages": response}


    def build_graph(self, checkpointer: MemorySaver = None):
        '''
        Build the state graph for the agent.
        Args:
            checkpointer (MemorySaver): Optional checkpointing mechanism.
        Returns:
            StateGraph: The constructed state graph.
        '''
        builder = StateGraph(State)

        # Define nodes
        builder.add_node("assistant", self.call_model)
        builder.add_node("tools", ToolNode(self.tools))
        # Define edges
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")

        graph = builder.compile(checkpointer=checkpointer)

        return graph
    

app = FastAPI()


@app.get("/")
def read_root():
    '''
    Root endpoint for the FastAPI application.
    Returns:
        dict: A welcome message.
    '''
    return {"Message": "Welcome to test-task!"}


@app.post("/ask/")
async def ask(question: Question):
    '''
    Endpoint to handle questions and return answers.
    Args:
        question (Question): The question to ask.
    Returns:
        dict: The question, answer, context, and response metadata.
    '''
    # Initialize the LLM and embedding model
    llm = AzureChatOpenAI(
        api_version="2024-10-21",
        azure_deployment="gpt-4o"
    )
    agent = QAAgent(llm)
    # Initialize the graph
    graph = agent.build_graph()
    response = graph.invoke({"messages": [HumanMessage(content=question.question)]})
    question = question.question
    answer = response.get("messages")[-1].content
    response_metadata = response.get("messages")[-1].response_metadata

    if len(response.get("messages")) > 2 and isinstance(response.get("messages")[2], ToolMessage):
        context = response.get("messages")[2].content
    else:
        context = "Out-of-scope question."

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "response_metadata": response_metadata
    }
