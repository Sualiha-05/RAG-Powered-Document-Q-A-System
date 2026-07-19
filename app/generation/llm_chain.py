# from langchain_community.llms import LlamaCpp
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from app.config import LLM_MODEL_PATH, LLM_N_CTX, LLM_N_THREADS, LLM_MAX_TOKENS, LLM_TEMPERATURE

# PROMPT_TEMPLATE = """Use the following context to answer the question.
# If the answer isn't in the context, say you don't know — don't make one up.

# Context:
# {context}

# Question: {question}

# Answer:"""

# def load_llm():
#     return LlamaCpp(
#         model_path=LLM_MODEL_PATH,
#         n_ctx=LLM_N_CTX,
#         n_threads=LLM_N_THREADS,
#         max_tokens=LLM_MAX_TOKENS,
#         temperature=LLM_TEMPERATURE,
#         stop=["Question:", "\nQuestion"],
#         verbose=False,
#     )

# def build_qa_chain(retriever):
#     llm = load_llm()
#     prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",
#         chain_type_kwargs={"prompt": prompt},
#         return_source_documents=True,
#     )

import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = """Use the following context to answer the question.
If the answer isn't in the context, say you don't know — don't make one up.

Context:
{context}

Question: {question}

Answer:"""


def load_llm():
    return ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=256,
    )


def build_qa_chain(retriever):
    llm = load_llm()
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )