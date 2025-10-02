import streamlit as st
from PIL import Image
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import re
import logging
import pytesseract
import pdfplumber
from pdf2image import convert_from_bytes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Tesseract path (adjust for your system)
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
    logger.error("Tesseract path configuration failed")

# State definition for LangGraph
class MedicalAgentState(TypedDict):
    image: Any
    extracted_text: str
    llm_response: str
    error: str

# Initialize Ollama LLM
llm_cpu = OllamaLLM(model="phi3:3.8b-mini-4k-instruct-q4_K_M", num_gpu=0)

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    You are a medical expert AI. Below is the extracted text from a medical test report. 
    Analyze the text, interpret the results, and provide a clear, concise explanation of the health status for a non-medical user. 
    Highlight any abnormal results, explain their potential implications, and suggest next steps (e.g., consult a doctor). 
    If any information is unclear or incomplete, note it and avoid making assumptions. define the origin of each terms completely without more details.

    Extracted Text:
    {extracted_text}

    Response format:
    **Health Status Summary**:
    [Your summary here]

    **Abnormal Results** (if any):
    [List abnormal results with explanations]

    **Recommendations**:
    [Next steps or advice]
    """
)

# Function to extract text from PDF or image
from io import BytesIO

def extract_text_from_file(file_bytes, filename: str) -> str:
    text = ""
    POPPLER_PATH = r"D:\poppler-25.07.0\Library\bin"  # adjust to your Poppler bin folder

    # Read the uploaded file bytes once
    file_bytes.seek(0)
    file_content = file_bytes.read()
    file_stream = BytesIO(file_content)  # wrap in BytesIO for pdfplumber

    if filename.lower().endswith(".pdf"):
        try:
            # Digital text extraction
            with pdfplumber.open(file_stream) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # Fallback to OCR if no text
            if not text.strip():
                logger.info("No text in PDF, falling back to OCR...")
                images = convert_from_bytes(file_content, dpi=150, poppler_path=POPPLER_PATH)
                for img in images:
                    text += pytesseract.image_to_string(img, lang="eng") + "\n"

        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            text = ""

    else:
        try:
            image_stream = BytesIO(file_content)
            image = Image.open(image_stream)
            text = pytesseract.image_to_string(image, lang="eng")
        except Exception as e:
            logger.error(f"Image OCR failed: {str(e)}")
            text = ""

    return text.strip()


# Node for LangGraph
def extract_text_from_image(state: dict) -> dict:
    def clean_ocr_text(raw_text: str) -> str:
        cleaned = re.sub(r"\s+", " ", raw_text).strip()
        cleaned = re.sub(r"[^\w\s\%\.\-\[\]/]+", "", cleaned)
        return cleaned

    try:
        uploaded_file = state["image"]
        filename = getattr(uploaded_file, "name", "uploaded_file")
        extracted_text = extract_text_from_file(uploaded_file, filename)
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

# Chain LLM with prompt
chain = prompt_template | llm_cpu | StrOutputParser()

# Node to process LLM
def process_with_llm(state: MedicalAgentState) -> MedicalAgentState:
    if state.get("error"):
        return state
    try:
        logger.info("Processing text with LLM...")
        cleaned_text = state["extracted_text"]
        response = chain.invoke({"extracted_text": cleaned_text})
        state["llm_response"] = response
        logger.info("LLM processing completed")
    except Exception as e:
        state["error"] = f"Error during LLM processing: {str(e)}"
        logger.error(f"LLM processing failed: {str(e)}")
    return state

# Build LangGraph workflow
def build_graph():
    workflow = StateGraph(MedicalAgentState)
    workflow.add_node("extract_text", extract_text_from_image)
    workflow.add_node("process_llm", process_with_llm)
    workflow.add_edge("extract_text", "process_llm")
    workflow.add_edge("process_llm", END)
    workflow.set_entry_point("extract_text")
    return workflow.compile()

# Streamlit UI
def main():
    st.title("Medical Test Report Analyzer")
    st.write("Upload a medical test report (image or PDF) to extract and interpret the results.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "pdf"])
    
    if uploaded_file is not None:
        # Display image or first page of PDF
        if uploaded_file.name.lower().endswith(".pdf"):
            first_page = convert_from_bytes(uploaded_file.read())[0]
            st.image(first_page, caption="Uploaded PDF (first page)", use_container_width=True)
        else:
            st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)
        
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
                if "Tesseract OCR" in result["error"]:
                    st.info("Note: Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")
            else:
                st.subheader("Extracted Text")
                st.text_area("Text from report", result["extracted_text"], height=200)
                
                st.subheader("Health Status Interpretation")
                st.markdown(result["llm_response"])
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Workflow execution failed: {str(e)}")

if __name__ == "__main__":
    main()
