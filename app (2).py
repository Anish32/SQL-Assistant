import streamlit as st
import pandas as pd
import openai
import google.generativeai as genai
from datetime import datetime
import os

# ✅ OpenAI client
client = openai.OpenAI(api_key="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")

# ✅ Gemini config
gemini_api_key = "AIzaSyBiAWj65GVR5sAzeo6ngCRb7X6qHBSIPPA"
genai.configure(api_key=gemini_api_key)

# ---- Local file to save results ----
OUTPUT_FILE = "output.csv"
if not os.path.exists(OUTPUT_FILE):
    pd.DataFrame(columns=["Timestamp", "Model", "Question", "SQL", "Explanation"]).to_csv(OUTPUT_FILE, index=False)

# ---- SQL Generator Functions ----
def generate_sql_openai(question):
    prompt = f"Convert this question to SQL:\n{question}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert SQL assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def explain_sql_openai(sql_query):
    prompt = f"Explain this SQL query in simple terms:\n{sql_query}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who explains SQL queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def generate_sql_gemini(question):
    prompt = f"""
You are an expert SQL assistant. Convert this question into a SQL query:

{question}

Only output the SQL code. No explanation or formatting.
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip().replace("```sql", "").replace("```", "").strip()

def explain_sql_gemini(sql_query):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Explain this SQL query in simple terms:\n{sql_query}"
    response = model.generate_content(prompt)
    return response.text.strip()

# ---- Save to CSV ----
def save_to_file(model, question, sql, explanation):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([[timestamp, model, question, sql, explanation]], columns=["Timestamp", "Model", "Question", "SQL", "Explanation"])
    row.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

# ---- Streamlit UI ----
st.set_page_config(page_title="🧠 Multi-LLM SQL Generator")
st.title("🧠 Multi-LLM SQL Generator ")

st.write("Ask a question in plain English, select your AI model, and get SQL + explanation.")

question = st.text_input("Enter your question in English")

model_choice = st.selectbox("Choose the AI model", ["Gemini 1.5 Flash", "OpenAI GPT-4"])

if st.button("Generate SQL & Explanation"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating SQL..."):
            if model_choice == "Gemini 1.5 Flash":
                sql = generate_sql_gemini(question)
                explanation = explain_sql_gemini(sql)
                model_used = "Gemini 1.5 Flash"
            elif model_choice == "OpenAI GPT-4":
                sql = generate_sql_openai(question)
                explanation = explain_sql_openai(sql)
                model_used = "OpenAI GPT-4"
            else:
                st.error("Unsupported model selected.")
                st.stop()

        st.subheader("Generated SQL:")
        st.code(sql, language="sql")

        st.subheader("SQL Explanation:")
        st.write(explanation)

        save_to_file(model_used, question, sql, explanation)
        st.success("✅ Output saved to output.csv")
