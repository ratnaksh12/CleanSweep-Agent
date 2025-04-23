import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def generate_action_plan(df: pd.DataFrame) -> str:
    chat = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-70b-8192"
    )

    sample = df.head(10).to_csv(index=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a senior data scientist mentoring a junior analyst. "
         "You are given a dataset and you need to suggest the next steps to extract maximum value from it."
        ),
        ("human", 
         "Here is a sample of the dataset:\n\n{sample}\n\n"
         "Based on this, suggest a detailed action plan. "
         "Include ideas for feature engineering, target column selection, ML use cases, EDA (exploratory data analysis), visualization ideas, and questions that can be answered using this data."
        )
    ])

    chain = prompt | chat
    result = chain.invoke({"sample": sample})
    return result.content
