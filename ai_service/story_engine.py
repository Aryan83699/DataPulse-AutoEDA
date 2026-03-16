import os
import pandas as pd
import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm(max_tokens=1000):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.5,
        max_tokens=max_tokens,
    )


def generate_data_story(csv_path: str) -> dict:
    """
    Generates:
    - domain: what field this data belongs to
    - features: one-line explanation per column
    - story: 2 short paragraphs about the dataset
    """
    try:
        if csv_path.endswith(".csv"):
            df = pd.read_csv(csv_path)
        else:
            df = pd.read_excel(csv_path)
    except Exception as e:
        return {"error": f"Could not load dataset: {str(e)}"}

    # Build column summary for the prompt
    col_summaries = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(2).tolist()
        null_pct = round(df[col].isnull().mean() * 100, 1)
        col_summaries.append(
            f"- {col} (type: {dtype}, nulls: {null_pct}%, sample: {sample})"
        )

    col_text = "\n".join(col_summaries)
    n_rows, n_cols = df.shape

    # Dynamic token limit based on column count
    max_tokens = min(500 + (n_cols * 30), 1500)

    system_prompt = """You are a concise data storyteller.
Respond ONLY in this exact JSON format (no markdown, no extra text):
{
  "domain": "one sentence about the field/industry",
  "features": {
    "column_name": "one line max per column",
    ...
  },
  "story": "exactly 2 short paragraphs about the dataset"
}"""

    user_prompt = f"""Dataset has {n_rows} rows and {n_cols} columns.

Columns:
{col_text}

Generate the data story JSON."""

    try:
        llm = get_llm(max_tokens=max_tokens)

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        raw = response.content.strip()

        # Clean markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Try direct parse first
        try:
            result = json.loads(raw)
            return result
        except json.JSONDecodeError:
            pass

        # Extract JSON block using regex
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                return result
            except json.JSONDecodeError:
                pass

        # Last resort — retry with stricter prompt
        strict_response = llm.invoke([
            SystemMessage(content="You are a JSON generator. Output ONLY valid JSON, nothing else. No explanation, no markdown."),
            HumanMessage(content=user_prompt + "\n\nIMPORTANT: Return ONLY the JSON object. No extra text.")
        ])

        raw2 = strict_response.content.strip()
        raw2 = raw2.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw2)
        return result

    except Exception as e:
        return {"error": f"Story generation failed: {str(e)}"}