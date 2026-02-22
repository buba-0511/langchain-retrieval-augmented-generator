import os
from dotenv import load_dotenv
import bs4
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def run_rag_project():
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    index_name = os.getenv("PINECONE_INDEX_NAME")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = PineconeVectorStore.from_documents(
        splits, 
        embeddings, 
        index_name=index_name
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Definimos el prompt para RAG
    system_prompt = (
        "Eres un asistente para tareas de preguntas y respuestas. "
        "Usa los siguientes fragmentos de contexto recuperado para responder la pregunta. "
        "Si no sabes la respuesta, di que no lo sabes. "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Construir la cadena de recuperación
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

    # 5. EJECUCIÓN
    response = rag_chain.invoke({"input": "¿What is task decomposition?"})
    
    print("\n--- Respuesta RAG ---")
    print(response["answer"])

if __name__ == "__main__":
    run_rag_project()