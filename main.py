# main.py (Test Version for Local + Railway)
import streamlit as st
from PIL import Image, ImageFilter
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import logging
import pytesseract
import pdfplumber
from pdf2image import convert_from_bytes
from io import BytesIO
import platform
import os

# Optional: Ollama LLM only if running locally
try:
    from langchain_ollama import OllamaLLM
    ollama_available = True
except ImportError:
    ollama_available = False

# ------------------------
# Logging
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# Cross-platform Tesseract & Poppler paths
# ------------------------
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH = r"D:\poppler-25.07.0\Library\bin"
else:
    # Linux / Docker
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    POPPLER_PATH = "/usr/bin"

# ------------------------
# LangGraph state
# ------------------------
class MedicalAgentState(TypedDict):
    image: Any
    extracted_text: str
    llm_response: str
    error: str

# ------------------------
# Initialize LLM only if available and local
# ------------------------
USE_LLM = ollama_available and platform.system() == "Windows"
if USE_LLM:
    llm_cpu = OllamaLLM(model="phi3:3.8b-mini-4k-instruct-q4_K_M", num_gpu=0)

prompt_template = ChatPromptTemplate.from_template("""
You are a medical expert AI. Below is the extracted text from a medical test report. 
Analyze the text, interpret the results, and provide a clear, concise explanation of the health status for a non-medical user. 
Highlight any abnormal results, explain their potential implications, and suggest next steps (e.g., consult a doctor). 
If any information is unclear or incomplete, note it and avoid making assumptions. define the origin of each term completely without more details.

Extracted Text:
{extracted_text}

Response format:
**Health Status Summary**:
[Your summary here]

**Abnormal Results** (if any):
[List abnormal results with explanations]

**Recommendations**:
[Next steps or advice]
""")

# ------------------------
# OCR + PDF extraction
# ------------------------
def extract_text_from_file(file_bytes: BytesIO, filename: str) -> str:
    text = ""
    file_bytes.seek(0)
    content = file_bytes.read()
    stream = BytesIO(content)

    if filename.lower().endswith(".pdf"):
        try:
            # Digital text extraction first
            with pdfplumber.open(stream) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # Fallback to OCR if no text
            if not text.strip():
                logger.info("No digital text found, using OCR on PDF pages...")
                images = convert_from_bytes(content, dpi=200, poppler_path=POPPLER_PATH)
                for img in images:
                    gray = img.convert("L").filter(ImageFilter.SHARPEN)
                    text += pytesseract.image_to_string(gray, lang="eng") + "\n"

        except Exception as e:
            logger.error("PDF extraction failed: %s", e)
            text = ""

    else:
        try:
            img = Image.open(stream).convert("L").filter(ImageFilter.SHARPEN)
            text = pytesseract.image_to_string(img, lang="eng")
        except Exception as e:
            logger.error("Image OCR failed: %s", e)
            text = ""

    logger.info(f"OCR Extracted Text:\n{text[:500]}...")  # log first 500 chars
    return text.strip()

# ------------------------
# LangGraph nodes
# ------------------------
def clean_ocr_text(raw_text: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw_text).strip()
    cleaned = re.sub(r"[^\w\s\%\.\-\[\]/]+", "", cleaned)
    return cleaned

def extract_text_from_image(state: dict) -> dict:
    try:
        uploaded_file = state["image"]
        filename = getattr(uploaded_file, "name", "uploaded_file")
        file_bytes = BytesIO(uploaded_file.read())

        extracted_text = extract_text_from_file(file_bytes, filename)
        cleaned_text = clean_ocr_text(extracted_text)

        if not cleaned_text:
            state["error"] = "No text could be extracted from the file."
            logger.warning("No text extracted from file")
        else:
            state["extracted_text"] = cleaned_text
            logger.info("Text extracted successfully")

    except Exception as e:
        state["error"] = f"Error during text extraction: {str(e)}"
        logger.error(f"Text extraction failed: {str(e)}")

    return state

# ------------------------
# LLM node
# ------------------------
if USE_LLM:
    chain = prompt_template | llm_cpu | StrOutputParser()
    def process_with_llm(state: MedicalAgentState) -> MedicalAgentState:
        if state.get("error"):
            return state
        try:
            response = chain.invoke({"extracted_text": state["extracted_text"]})
            state["llm_response"] = response
        except Exception as e:
            state["error"] = f"LLM processing failed: {str(e)}"
        return state
else:
    # Dummy LLM for Railway / cloud
    def process_with_llm(state: MedicalAgentState) -> MedicalAgentState:
        state["llm_response"] = "LLM skipped: only OCR tested (server cannot reach local Ollama)."
        return state

# ------------------------
# Build LangGraph workflow
# ------------------------
def build_graph():
    workflow = StateGraph(MedicalAgentState)
    workflow.add_node("extract_text", extract_text_from_image)
    workflow.add_node("process_llm", process_with_llm)
    workflow.add_edge("extract_text", "process_llm")
    workflow.add_edge("process_llm", END)
    workflow.set_entry_point("extract_text")
    return workflow.compile()

# ------------------------
# Streamlit UI
# ------------------------
def main():
    st.title("Medical Test Report Analyzer")
    st.write("Upload a medical test report (image or PDF) to extract and interpret the results.")

    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file:
        uploaded_file.seek(0)
        file_bytes = BytesIO(uploaded_file.read())
        uploaded_file.seek(0)

        # Display image or PDF first page
        if uploaded_file.name.lower().endswith(".pdf"):
            images = convert_from_bytes(file_bytes.read(), dpi=150, poppler_path=POPPLER_PATH)
            st.image(images[0], caption="Uploaded PDF (first page)", use_container_width=True)
        else:
            st.image(Image.open(file_bytes), caption="Uploaded Image", use_container_width=True)

        # Initialize state
        state = MedicalAgentState(
            image=uploaded_file,
            extracted_text="",
            llm_response="",
            error=""
        )

        # Run workflow
        try:
            graph = build_graph()
            result = graph.invoke(state)

            if result.get("error"):
                st.error(result["error"])
            else:
                st.subheader("Extracted Text")
                st.text_area("Text from report", result["extracted_text"], height=200)

                st.subheader("Health Status Interpretation")
                st.markdown(result["llm_response"])

        except Exception as e:
            st.error(f"Workflow execution failed: {str(e)}")
            logger.error(f"Workflow execution failed: {str(e)}")

if __name__ == "__main__":
    main()
