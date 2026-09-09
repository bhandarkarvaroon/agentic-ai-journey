# day55.py — Chat with your PDF using Streamlit + RAG

import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import tempfile

st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and st.session_state.vectorstore is None:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            st.session_state.chunks_count = len(chunks)

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY")
            )
            st.session_state.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="pdf_chat"
            )
            os.unlink(tmp_path)
        st.success(f"PDF processed! {st.session_state.chunks_count} chunks indexed.")

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.rerun()

if st.session_state.vectorstore is None:
    st.info("Please upload a PDF to start chatting.")
else:
    st.success(f"PDF loaded — {st.session_state.chunks_count} chunks ready.")

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer based ONLY on the context below.
Be specific and detailed. Quote directly from the document when relevant.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}")
    ])

    def format_docs(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        else:
            with st.chat_message("assistant"):
                st.write(msg.content)

    if question := st.chat_input("Ask a question about your PDF..."):
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Create retriever fresh from session state
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                )
                retrieved_docs = retriever.invoke(question)
                context = format_docs(retrieved_docs)

                # Debug — show what was retrieved
                with st.expander("Retrieved context"):
                    st.write(context)

                answer_chain = (
                    answer_prompt
                    | llm
                    | StrOutputParser()
                )
                answer = answer_chain.invoke({
                    "context": context,
                    "question": question,
                    "chat_history": st.session_state.chat_history
                })
            st.write(answer)

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))
