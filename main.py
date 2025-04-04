import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import os

# Rebuild model schema
ChatOllama.model_rebuild()

# --- Define Specialist Tools ---
@tool
def analyze_cardiologist(medical_report: str):
    """
    Analyze the patient's report from a cardiologist's perspective.
    """
    prompt = f"""
    Act like a cardiologist. You will receive a medical report of a patient.
    Task: Review the patient's cardiac workup, including ECG, blood tests, Holter monitor results, and echocardiogram.
    Focus: Determine if there are any subtle signs of cardiac issues that could explain the patient's symptoms. Rule out any underlying heart conditions.
    Recommendation: Provide guidance on any further cardiac testing or monitoring needed. Suggest potential management strategies.
    Medical Report: {medical_report}
    """
    llm = ChatOllama(model="mistral-nemo:latest", temperature=0.3)
    return llm.invoke(prompt).content

@tool
def analyze_psychologist(medical_report: str):
    """
    Analyze the patient's report from a psychologist's perspective.
    """
    prompt = f"""
    Act like a psychologist. You will receive a patient's report.
    Task: Review the report and provide a psychological assessment.
    Focus: Identify potential mental health issues such as anxiety or depression.
    Recommendation: Offer guidance on therapy or other interventions.
    Medical Report: {medical_report}
    """
    llm = ChatOllama(model="mistral-nemo:latest", temperature=0.3)
    return llm.invoke(prompt).content

@tool
def analyze_pulmonologist(medical_report: str):
    """
    Analyze the patient's report from a pulmonologist's perspective.
    """
    prompt = f"""
    Act like a pulmonologist. You will receive a patient's report.
    Task: Review the report and provide a pulmonary assessment.
    Focus: Identify potential respiratory issues such as asthma, COPD, or infections.
    Recommendation: Offer guidance on tests or treatments.
    Medical Report: {medical_report}
    """
    llm = ChatOllama(model="mistral-nemo:latest", temperature=0.3)
    return llm.invoke(prompt).content

@tool
def analyze_multidisciplinary(cardiologist_report: str, psychologist_report: str, pulmonologist_report: str):
    """
    Combine specialist reports to determine three possible health issues.
    """
    prompt = f"""
    You are a multidisciplinary team of healthcare professionals.
    Task: Review the cardiologist, psychologist, and pulmonologist reports.
    Output: List 3 possible health issues the patient may have, with reasons for each.

    Cardiologist Report: {cardiologist_report}
    Psychologist Report: {psychologist_report}
    Pulmonologist Report: {pulmonologist_report}
    """
    llm = ChatOllama(model="mistral-nemo:latest", temperature=0.3)
    return llm.invoke(prompt).content

# --- Streamlit App UI ---
st.set_page_config(page_title="Multidisciplinary Medical Analyzer", page_icon="🧠")
st.title("🧠 Multidisciplinary Medical Report Analyzer")

uploaded_file = st.file_uploader("📤 Upload a medical report (TXT)", type=["txt"])

if uploaded_file:
    medical_report = uploaded_file.read().decode("utf-8")

    st.markdown("### 🧑‍⚕️ Choose a Specialist to Analyze the Report")
    specialist = st.selectbox(
        "Select Specialist",
        options=["Cardiologist", "Psychologist", "Pulmonologist", "All Specialists"]
    )

    if st.button("🔍 Analyze Report"):
        with st.spinner("Analyzing report..."):

            if specialist == "Cardiologist":
                result = analyze_cardiologist.invoke({"medical_report": medical_report})
                st.markdown("### 🫀 Cardiologist Report")
                st.text(result)
            elif specialist == "Psychologist":
                result = analyze_psychologist.invoke({"medical_report": medical_report})
                st.markdown("### 🧠 Psychologist Report")
                st.text(result)
            elif specialist == "Pulmonologist":
                result = analyze_pulmonologist.invoke({"medical_report": medical_report})
                st.markdown("### 🌬️ Pulmonologist Report")
                st.text(result)
            elif specialist == "All Specialists":
                st.markdown("### ⏳ Running all specialist tools...")
                cardiologist_report = analyze_cardiologist.invoke({"medical_report": medical_report})
                psychologist_report = analyze_psychologist.invoke({"medical_report": medical_report})
                pulmonologist_report = analyze_pulmonologist.invoke({"medical_report": medical_report})

                st.success("✅ All specialist reports generated!")

                with st.expander("🫀 Cardiologist Report"):
                    st.text(cardiologist_report)
                with st.expander("🧠 Psychologist Report"):
                    st.text(psychologist_report)
                with st.expander("🌬️ Pulmonologist Report"):
                    st.text(pulmonologist_report)

                # Final multidisciplinary analysis
                with st.spinner("🔍 Combining results for final diagnosis..."):
                    final_diagnosis = analyze_multidisciplinary.invoke({
                        "cardiologist_report": cardiologist_report,
                        "psychologist_report": psychologist_report,
                        "pulmonologist_report": pulmonologist_report
                    })

                st.markdown("### ✅ Final Diagnosis")
                st.markdown(final_diagnosis)

                # Save to file
                output_path = "results/final_diagnosis.txt"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(final_diagnosis)

                st.success(f"📁 Final diagnosis saved to `{output_path}`.")
