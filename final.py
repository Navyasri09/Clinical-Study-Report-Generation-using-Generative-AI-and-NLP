from datetime import datetime
from altair import Self
import numpy as np
import pandas as pd
import streamlit as st
import asyncio
from interact.base import Cascade, Handler, Message
from interact.handlers import OpenAiLLM
from lookup import Lookup


st.title("Clinical Study Report")

# Load Dataset
df = pd.read_csv("Patientnarrative.csv")
data = pd.read_csv("Disease.csv")

# User Input
min_date = datetime(1950, 1, 1)
col1, col2 = st.columns(2)
with col1:
    patient_id = st.text_input("Enter Patient ID:")
with col2:
    doa = st.date_input(
        "Enter Date of Admission:", format="DD-MM-YYYY", min_value=min_date
    )


# Filtering data
df = df[df["Patient ID"].astype(str) == patient_id]
# df = df[df["Date of admission"] == doa]


# Handler for dm
class DemographicsPrompt(Handler):
    role = "DemographicsGenerator"
    prompt = "Given a data about patient which contains patient information and his clinical conditions. Your task is to Summarize the data into Demographics, Medical History, Clinical Condition.\nFor example: Demographics contains patient id, sex, age, etc..\nMedical history contains whether patient has prior chronic disease, etc.. \n Clinical conditions include heamoglobin, glucose and all his vitals.\n Note:\n1. Represent Medical history in form of Paragraph.\n2. In Clinical conditions, if absent, don't include.\n Data: {data}."

    async def process(self, msg: Message, csd: Cascade) -> str:
        data = msg.primary
        prompt = self.prompt.format(data=data.to_markdown())

        return prompt


# Handler for medical history
class MedicalPrompt(Handler):
    role = "MedicalHistoryGenerator"
    prompt = "Given data about patient. Your task is to only display the terms patient is suffering from in specified format. It can be smoking, alcohol, any prior disease. If it has 1, include it in medical terms. If there are more than one 1's include all.\n Example: \nMedical terms:\n1. Stable Agina\n2. Kidney Disease.\n Note: Just display the disease. If anything is absent, don't include.\n Data: {data}"

    async def process(self, msg: Message, csd: Cascade) -> str:
        data = msg.primary
        prompt = self.prompt.format(data=data.to_markdown())

        return prompt


# Handler for Report
class ReportPrompt(Handler):
    role = "ReportGenerator"
    prompt = "Given medical terms of patient separated by comma(','). Just elaborate that patient is suffering from these terms. in a simple paragraph.\nNote: Hihlight the medical terms in paragraph by keeping them in Bold and Capital Letters.\n Medical Terms: {data}"

    async def process(self, msg: Message, csd: Cascade) -> str:
        data = msg.primary
        prompt = self.prompt.format(data=data)

        return prompt


demo_generator = DemographicsPrompt() >> OpenAiLLM()
medical_generator = MedicalPrompt() >> OpenAiLLM()
report_generator = ReportPrompt() >> OpenAiLLM()

similarity_finder = Lookup(data)

temp = st.button("Generate")
if temp:
    st.write("Reference Data:")
    st.table(df)

    result1 = asyncio.run(demo_generator.start(Message(df)))
    st.write(result1.last_msg.primary)

    result2 = asyncio.run(medical_generator.start(Message(df)))
    medical_history = result2.last_msg.primary
    

    final = ""
    required_info_start_index = medical_history.lower().find("medical terms")
    required_info_lines = medical_history[required_info_start_index:].split("\n")[1:]
    for line in required_info_lines:
        most_similar_variable = similarity_finder.most_similar_column(line)
        final = final + most_similar_variable + ","

    result3 = asyncio.run(report_generator.start(Message(final)))
    st.write(result3.last_msg.primary)
