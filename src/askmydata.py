# src/ask_my_data.py
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

def ask_data(query: str, df: pd.DataFrame) -> str:
    """
    Use Groq Llama3 to answer user questions about the marketing dataset.
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Groq API key not found. Please set GROQ_API_KEY in your environment variables."

    # Load the LLM
    llm = ChatGroq(model=os.getenv("GROQ_MODEL"), temperature=0.3)


    # Create a textual summary of dataset stats (so the LLM has context)
    stats = df.describe(include="all").to_string()

    prompt = PromptTemplate(
        input_variables=["query", "stats"],
        template=(
            "You are a data analyst assistant. You are analyzing a marketing campaign dataset. "
            "Use the following dataset statistics to answer the question concisely and accurately.\n\n"
            "Dataset Summary:\n{stats}\n\n"
            "Question: {query}\n\n"
            "Answer clearly and in plain language, focusing on insights from the data."
        ),
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(query=query, stats=stats)
    return response.strip()
