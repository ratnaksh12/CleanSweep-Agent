import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from scipy import stats
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 🔍 Detect outliers based on Z-score
def detect_outliers(df):
    outlier_suggestions = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        z_scores = stats.zscore(df[column].dropna())
        outliers = np.where(np.abs(z_scores) > 3)
        if len(outliers[0]) > 0:
            outlier_suggestions[column] = f"Outliers detected in '{column}' (Z-score > 3). Consider removing or capping extreme values."
    return outlier_suggestions

# 🤖 Get AI-generated cleaning suggestions with outlier insight
def get_cleaning_suggestions(df):
    chat = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-8b-8192"
    )

    sample = df.head(10).to_csv(index=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior data analyst who gives cleaning suggestions to junior analysts."),
        ("human", 
         "Here are the first few rows of a dataset:\n\n{sample}\n\n"
         "Give a list of suggestions to clean this data for better analysis. "
         "Suggestions can include renaming columns, handling nulls, dropping irrelevant columns, removing duplicates, fixing data types, etc. "
         "Make sure it's helpful, practical, and clear.")
    ])

    chain = prompt | chat
    result = chain.invoke({"sample": sample})

    # Include outlier detection suggestions
    outliers = detect_outliers(df)
    if outliers:
        result.content += "\n\n[📊 Outlier Suggestions]\n"
        result.content += "\n".join([f"- {msg}" for msg in outliers.values()])

    return result.content

# ✅ Automatically apply cleaning suggestions and return change logs
def apply_cleaning_suggestions(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    cleaned_df = df.copy()
    changes = []

    # 1. Drop duplicates
    before = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    after = len(cleaned_df)
    if before != after:
        changes.append(f"Removed {before - after} duplicate rows.")

    # 2. Drop columns with >50% missing values
    cols_before = set(cleaned_df.columns)
    threshold = 0.5 * len(cleaned_df)
    cleaned_df = cleaned_df.dropna(axis=1, thresh=threshold)
    dropped_cols = cols_before - set(cleaned_df.columns)
    if dropped_cols:
        changes.append(f"Dropped columns with >50% missing values: {', '.join(dropped_cols)}.")

    # 3. Fill remaining nulls
    for col in cleaned_df.columns:
        nulls = cleaned_df[col].isnull().sum()
        if nulls > 0:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                changes.append(f"Filled {nulls} nulls in numeric column '{col}' with mean.")
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0])
                changes.append(f"Filled {nulls} nulls in non-numeric column '{col}' with mode.")

    # 4. Rename columns to snake_case
    old_columns = cleaned_df.columns
    new_columns = [col.strip().lower().replace(" ", "_") for col in old_columns]
    if list(old_columns) != new_columns:
        changes.append("Renamed columns to snake_case.")
    cleaned_df.columns = new_columns

    return cleaned_df, changes
