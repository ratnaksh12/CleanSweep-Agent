import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def get_dataset_overview(df):
    chat = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-8b-8192"
    )

    sample = df.head(10).to_csv(index=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a helpful data analyst."),
        ("human", "Here's the first few rows of a dataset:\n\n{sample}\n\nWhat is this dataset likely about? Describe its key features, columns, and any assumptions.")
    ])

    chain = prompt | chat

    result = chain.invoke({"sample": sample})
    return result.content
