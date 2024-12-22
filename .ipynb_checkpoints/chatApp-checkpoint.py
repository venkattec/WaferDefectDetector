import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import uuid
from langchainFlow import get_answer

st.set_page_config(page_title="WaferGPT", page_icon="📖", layout="wide")

# Function to get the most recently edited file in a folder
def get_most_recent_file(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# Beautify user and chatbot messages for dark mode
def beautify_user_message(message):
    return f"""<div style='background-color:#333;padding:10px;border-radius:10px;margin:10px 0;color:#fff;'>
                <strong style="color:#4DB8FF;">You:</strong> {message}
               </div>"""


def beautify_ca_message(message):
    return f"""<div style='background-color:#444;padding:10px;border-radius:10px;margin:10px 0;color:#fff;'>
                <strong style="color:#FFD700;">AI-Assistant:</strong> {message}
               </div>"""


# Ensure directory exists for processed images
os.makedirs("processed_images", exist_ok=True)

# Streamlit app title
st.title("Wafer-Defect-Detector")
st.sidebar.header("Wafer-Defect-Detector")

# Initialize session state variables
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_npy' not in st.session_state:
    st.session_state.selected_npy = None

# Sidebar "About" section
st.sidebar.subheader("About")
st.sidebar.info(
    """
    **WaferGPT** is a specialized AI-driven tool designed to analyze semiconductor wafers for defects.
    Pick a wafer image and interact with our AI assistant to detect issues and gain insights on the type
    of defect, location of the defect, defect percentage and much more
    """
)

# Sidebar "How to Use" section
st.sidebar.subheader("How to Use")
st.sidebar.info(
    """
    1. Select the file from the **"Pick a wafer to examine"** dropdown.
    2. Ask your question in the chatbox.
    3. View the AI's analysis and the processed image.
    """
)

# Sidebar for LLM selection
st.sidebar.subheader("Select a LLM Model")
LLMs = ['Llava', 'GPT-4o']
selected_llm = st.sidebar.selectbox("Choose a LLM:", LLMs, index=0)
if selected_llm:
    st.session_state.selected_llm = selected_llm

# Sidebar for selecting .npy file
st.sidebar.header("Pick a wafer to examine!!")
npy_files = [f for f in os.listdir('npy_files') if f.endswith('.npy')]
npy_file_selected = st.sidebar.selectbox("Choose a wafer:", npy_files)

if npy_file_selected:
    # Load the selected .npy file
    array = np.load(os.path.join('npy_files', npy_file_selected), allow_pickle=True)
    st.session_state.selected_npy = os.path.join('npy_files', npy_file_selected)
    st.sidebar.write(f"Selected wafer:")

    # Display the .npy file as an image in the sidebar
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(array, cmap='viridis')
    ax.axis('off')
    st.sidebar.pyplot(fig)

# Display chat history with image and question
for idx, message in enumerate(st.session_state.chat_history):
    if message['type'] == 'user':
        st.markdown(beautify_user_message(message['content']), unsafe_allow_html=True)
    elif message['type'] == 'ca':
        st.markdown(beautify_ca_message(message['content'].capitalize()), unsafe_allow_html=True)
        if 'image_path' in message:
            st.image(message['image_path'], caption="Processed Image", width=350)

# User input and selected .npy file interaction
user_prompt = st.chat_input("Ask your question here !!")

if user_prompt:
    # Record user question
    st.session_state.chat_history.append({'type': 'user', 'content': user_prompt})

    # Generate AI response
    response = get_answer(user_prompt, st.session_state.selected_npy)

    # Get the most recently edited file from the "output" folder
    recent_file = get_most_recent_file("output")
    if recent_file and response:
        # Generate a unique filename for the processed image
        unique_filename = f"processed_{uuid.uuid4().hex}.png"
        processed_image_path = os.path.join("processed_images", unique_filename)

        # Copy the most recent file to the processed_images folder with a new name
        shutil.copy(recent_file, processed_image_path)

        # Save AI response and processed image in chat history
        st.session_state.chat_history.append({
            'type': 'ca',
            'content': response,
            'image_path': processed_image_path
        })
    else:
        # If no file is found, inform the user
        st.session_state.chat_history.append({
            'type': 'ca',
            'content': response + "\nNo processed image available. Please ensure the 'output' folder contains images."
        })

    # Rerun to refresh the chat
    st.rerun()
