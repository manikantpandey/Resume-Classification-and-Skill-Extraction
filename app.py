import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import PyPDF2
import spacy
from bs4 import BeautifulSoup
import re

# Define category mapping
CATEGORY_MAPPING = {
    0: "HR", 1: "Designer", 2: "Information-Technology", 3: "Teacher", 4: "Advocate",
    5: "Business-Development", 6: "Healthcare", 7: "Fitness", 8: "Agriculture",
    9: "BPO", 10: "Sales", 11: "Consultant", 12: "Digital-Media", 13: "Automobile",
    14: "Chef", 15: "Finance", 16: "Apparel", 17: "Engineering", 18: "Accountant",
    19: "Construction", 20: "Public-Relations", 21: "Banking", 22: "Arts", 23: "Aviation"
}

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_path = "./tinyllama_resume_model_lora"  # Update this path
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    special_tokens_dict = {'additional_special_tokens': ['[CATEGORY]', '[RESUME]', '[END]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CATEGORY_MAPPING),
        load_in_8bit=True,
        device_map="auto"
    )
    
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    return tokenizer, model

tokenizer, model = load_model()

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def clean_html(html_text):
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', text).strip()

def process_text(text):
    if not text:
        return ""
    doc = nlp(text[:1000000])  # Limit to first 1M chars
    
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    skills = [token.text for token in doc if token.pos_ == "NOUN" and token.is_alpha]
    
    return f"Entities: {entities}\nSkills: {skills}\nOriginal: {text[:1000]}"

def classify_resume(text):
    processed_text = process_text(text)
    input_text = f"[CATEGORY][RESUME]{processed_text}[END]"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    return CATEGORY_MAPPING.get(predicted_class, "Unknown")

st.title("Resume Analyzer")

uploaded_file = st.file_uploader("Choose a resume file", type=["pdf", "txt", "html"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/html":
        resume_text = clean_html(uploaded_file.read().decode())
    else:
        resume_text = uploaded_file.getvalue().decode()
    
    st.subheader("Resume Text")
    st.text(resume_text[:500] + "...")  # Display first 500 characters
    
    job_category = classify_resume(resume_text)
    st.subheader("Job Category")
    st.write(f"Predicted Job Category: {job_category}")
    
    processed_text = process_text(resume_text)
    st.subheader("Processed Resume Information")
    st.text(processed_text)
    
    # Display category probabilities
    with torch.no_grad():
        inputs = tokenizer(f"[CATEGORY][RESUME]{processed_text}[END]", 
                           return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    st.subheader("Category Probabilities")
    for i, prob in enumerate(probabilities):
        st.write(f"{CATEGORY_MAPPING[i]}: {prob.item():.2%}")

# Add a sidebar with information about the app
st.sidebar.title("About")
st.sidebar.info("This app analyzes resumes, classifies job categories, extracts entities and skills, and provides probability scores for each category.")

# Add a footer
st.markdown("---")
