from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import (
    Language, RecursiveCharacterTextSplitter
)
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import logging
import time


def process_md_files():
    loader = GithubFileLoader(
        repo="vanna-ai/vanna",
        branch="main",
        access_token=os.getenv('GITHUB_ACCESS_TOKEN'),
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(".md"),
    )
    documents = loader.load()
    
    chunks = RecursiveCharacterTextSplitter.from_language(
        chunk_size=1000,
        chunk_overlap=200,
        language=Language.MARKDOWN
    ).split_documents(documents)

    logging.info(f"Loaded {len(documents)} markdown documents and split them into {len(chunks)} chunks.")

    return chunks


def process_python_code(docstring_model: AzureChatOpenAI):
    doc_string_sys_message = "You are a DocString and commentline generator. " \
    "You will be given a code snippet and you will generate a docstring and comment lines for the code. " \
    "Return the provided code snippet with the docstring and comment lines added. " \
    "The docstring should be in the format of a Python docstring. " \
    "The comment lines should be in the format of Python comments. "
    messages = [
        SystemMessage(content=doc_string_sys_message)
    ]

    python_loader = GithubFileLoader(
        repo="vanna-ai/vanna",
        branch="main",
        access_token=os.getenv('GITHUB_ACCESS_TOKEN'),
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(".py") and "__init__" not in file_path and "local" not in file_path and "remote" not in file_path and "utils" not in file_path and "assets" not in file_path and "test" not in file_path,
    )
    python_docs = python_loader.load()

    '''
    for idx, doc in enumerate(python_docs):
        page_content = doc.page_content
        commented_page_content = docstring_model.invoke(messages + [HumanMessage(content=page_content)]).content
        # Replace the original content with the commented code
        python_docs[idx].page_content = commented_page_content
    ''' # not used this time, but can be used to generate docstrings and comments for the code

    python_chunks = RecursiveCharacterTextSplitter.from_language(
        chunk_size=500,
        chunk_overlap=50,
        language=Language.PYTHON
    ).split_documents(python_docs)

    logging.info(f"Loaded {len(python_docs)} python documents and split them into {len(python_chunks)} chunks.")

    return python_chunks


def create_vector_index(documents, embedding_model: AzureOpenAIEmbeddings):
    vector_store = Chroma(
        collection_name="test-task-collection",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )

    vector_store.add_documents(documents)
    # Test vector index
    query = "What is the purpose of the Vanna project?"
    results = vector_store.similarity_search(query, k=5)
    logging.info(f"Query: {query}")
    logging.info(f"Results: {results}")


if __name__ == "__main__":
    load_dotenv()

    embedding_ada = AzureOpenAIEmbeddings(
        api_version="2024-10-21",
        azure_deployment="text-embedding-3-small-1"
    )

    docstring_model = gpt4o_mini_model = AzureChatOpenAI(
        api_version="2024-10-21",
        azure_deployment="gpt-4o-mini-2024-07-18",
        temperature=0,
        max_tokens=4000
    )

    md_documents = process_md_files()
    python_documents = process_python_code(docstring_model)
    documents = md_documents + python_documents
    logging.info(f"Total documents to be indexed: {len(documents)}")
    # Create vector index
    start_time = time.time()
    create_vector_index(documents, embedding_ada)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Vector index creation took {elapsed_time:.2f} seconds.")
    logging.info(f"Vector index creation took {elapsed_time:.2f} seconds.")
    logging.info("Vector index created successfully.")
    