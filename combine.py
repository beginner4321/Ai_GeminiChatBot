import streamlit as st
from PIL import Image
import google.generativeai as genai
import textwrap3
import os
from dotenv import load_dotenv

load_dotenv()  
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_image_response(input_text, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input_text != "":
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    
    if hasattr(response, 'parts') and response.parts:
        return response.text
    else:
        return "Sorry, the response is empty or invalid."

def get_gemini_question_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap3.indent(text, '> ', predicate=lambda _: True)

st.set_page_config(page_title="Gemini Demo")

options = ["Gemini Image Demo", "Q&A Demo"]
selected_option = st.sidebar.multiselect("Select Demo", options)

if "Gemini Image Demo" in selected_option:
    st.header("Gemini Image Demo")

    input_text = st.text_input("Input Prompt: ", key="input_image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = ""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    submit = st.button("Tell me about the image")
    if submit:
        response = get_gemini_image_response(input_text, image)
        st.subheader("The Response is")
        st.write(response)

if "Q&A Demo" in selected_option:
    st.header("Q&A Demo")

    input_question = st.text_input("Input: ", key="input_question")
    submit_question = st.button("Ask the question")
    if submit_question:
        response_question = get_gemini_question_response(input_question)
        st.subheader("The Response is")
        st.markdown(to_markdown(response_question))
