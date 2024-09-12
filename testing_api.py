# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_community.vectorstores import Chroma
# from langchain_nomic.embeddings import NomicEmbeddings
# from langchain_community.chat_models import ChatOllama
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_chroma import Chroma

# # Initialize FastAPI app
# app = FastAPI()

# # Define input for question
# class QueryRequest(BaseModel):
#     question: str

# # Load vector store
# # def load_vector_store():
# #     try:
# #         vectorstore = Chroma(
# #             collection_name="rag-chroma",
# #             embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
# #             persist_directory="./chroma_data"
# #         )
# #         return vectorstore.as_retriever()
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")
# from langchain.embeddings.openai import OpenAIEmbeddings  # Example for OpenAI embeddings

# def load_vector_store():
#     try:
#         # Initialize embeddings
#         embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        
#         # Load the vector store
#         vectorstore = Chroma(
#             persist_directory="./chroma_data",
#             collection_name="rag-chroma",
#         )
        
#         # Set up the retriever with embeddings
#         return vectorstore.as_retriever(embedding_function=embeddings)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")


# # Load LLM and prompt
# prompt = PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
#     Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
#     Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
#     Question: {question} 
#     Context: {context} 
#     Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
#     input_variables=["question", "context"],
# )

# llm = ChatOllama(model="llama3", temperature=0)
# rag_chain = prompt | llm | StrOutputParser()

# # Helper function to format documents
# # def format_docs(docs):
# #     return "\n\n".join(doc.page_content for doc in docs)

# # @app.post("/query")
# # async def query_vectorstore(request: QueryRequest):
# #     retriever = load_vector_store()
# #     docs = retriever.invoke(request.question)
    
# #     if docs:
# #         formatted_docs = format_docs(docs)
# #         generation = rag_chain.invoke({"context": formatted_docs, "question": request.question})
# #         return {"answer": generation}
# #     else:
# #         return {"answer": "No relevant information found in the documents."}
# @app.post("/query")
# async def query_vectorstore(request: QueryRequest):
#     try:
#         retriever = load_vector_store()
#         docs = retriever.invoke(request.question)
        
#         if docs:
#             formatted_docs = format_docs(docs)
#             generation = rag_chain.invoke({"context": formatted_docs, "question": request.question})
#             return {"answer": generation}
#         else:
#             return {"answer": "No relevant information found in the documents."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# # To run the app using Uvicorn, save this as api.py and run:
# # uvicorn api:app --reload







from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize FastAPI app
app = FastAPI()

# Define input for question
class QueryRequest(BaseModel):
    question: str

# Load vector store
def load_vector_store():
    try:
        # Initialize embeddings
        embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        
        # Load the vector store with the embeddings
        vectorstore = Chroma(
            persist_directory="./chroma_data",
            collection_name="rag-chroma",
            embedding_function=embeddings  # Pass the embeddings here
        )
        
        # Set up the retriever
        return vectorstore.as_retriever()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")

# Load LLM and prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)
llm = ChatOllama(model="llama3.1:70b", temperature=0)
rag_chain = prompt | llm | StrOutputParser()

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.post("/query")
async def query_vectorstore(request: QueryRequest):
    try:
        retriever = load_vector_store()
        docs = retriever.invoke(request.question)
        
        if docs:
            formatted_docs = format_docs(docs)
            generation = rag_chain.invoke({"context": formatted_docs, "question": request.question})
            return {"answer": generation}
        else:
            return {"answer": "No relevant information found in the documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# To run the app using Uvicorn, save this as api.py and run:
# uvicorn api:app --reload
