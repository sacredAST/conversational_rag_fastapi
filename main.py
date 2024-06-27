from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from pymongo import MongoClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

from app.schema import MessageRequest, ChatResponseModel

from pydantic import BaseModel
from typing import List

import numpy as np

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_models():
    global client, db, collection, llm, embeddings
    client = MongoClient(settings.ATLAS_URL)
    db = client[settings.DB_NAME]
    collection = db[settings.COLLECTION_NAME]
    llm = ChatOpenAI(model=settings.LLM_MODEL, api_key=settings.API_KEY, base_url=settings.API_URL)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cpu'}, encode_kwargs={'device': 'cpu'})

@app.on_event("shutdown")
async def shutdown_db():
    client.close()

@app.post("/api/save/knowledge")
async def save_knowledge(request):
    # loader = TextLoader("input.txt", "utf-8", True)

    # docs = loader.load()

    # text_spliter = RecursiveCharacterTextSplitter()

    # documents = text_spliter.split_documents(docs)

    # embeddings = HuggingFaceEmbeddings(model_name=settings.HUGGINGFACE_EMBEDDING,
    #                                        model_kwargs={'device': 'cpu'}, encode_kwargs={'device': 'cpu'})
    # vectorStore = MongoDBAtlasVectorSearch.from_documents(documents, embeddings, collection=collection, index_name=settings.INDEX_NAME)
    return

@app.post("/api/chat", response_model=ChatResponseModel)
async def chat(request: MessageRequest):
    vectorStore = MongoDBAtlasVectorSearch(collection, embeddings, index_name="embedding")
    retriever = vectorStore.as_retriever()

    # Prompt for generating search query from chat_history
    prompt = ChatPromptTemplate.from_messages(
        [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ]
    )

    # create retriever_chain based on current llm and search query
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    # prompt for generate answer the user's questions based on all history.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user'questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriever_chain, document_chain)

    chat_history = []

    for message in request.messages[:-1]:
        if message.role == "system":
            chat_history.append(AIMessage(content=message.content))
        elif message.role == "human":
            chat_history.append(HumanMessage(content=message.content))

    output = retriever_chain.invoke({
        "chat_history": chat_history,
        "input": request.messages[-1].content
    })

    response = ChatResponseModel(
        chatID=request.chatID,
        message=output['answer']
    )

    return response