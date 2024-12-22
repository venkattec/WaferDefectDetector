import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64


# Mock function to process image and text (simulate chatbot response)
def process_query(image, query):
    return f"Processed query: '{query}' with selected image."


# Function to render `.npy` image with Matplotlib
def render_npy_image(npy_array):
    fig, ax = plt.subplots()
    ax.imshow(npy_array, cmap="gray")  # Adjust colormap as needed
    ax.axis("off")  # Remove axes for cleaner display
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf


# Example `.npy` file paths (replace with actual paths)
npy_file_paths = ["image1.npy", "image2.npy", "image3.npy"]

# Load `.npy` files into arrays
images = [np.load(path) for path in npy_file_paths]

# Initialize Streamlit app
st.title("Picture-Based Chatbot")

# Sidebar for picture selection
st.sidebar.title("Select a Picture")

# Initialize session state for selection and conversation history
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []


# Function to handle image selection
def select_image(index):
    st.session_state.selected_image = index


# Display all images as clickable buttons in sidebar
for i, img_array in enumerate(images):
    img_buf = render_npy_image(img_array)
    img_data = img_buf.getvalue()

    # Convert the image to base64 for embedding in HTML
    img_base64 = base64.b64encode(img_data).decode("utf-8")
    is_selected = st.session_state.selected_image == i
    border_color = "blue" if is_selected else "transparent"

    # Display image with a border and make it clickable via button
    if st.sidebar.button(f"Select Picture {i + 1}", key=f"image_{i}"):
        select_image(i)

    # Display image with a border around selected image
    st.sidebar.image(img_buf, caption=f"Picture {i + 1}", use_column_width=True)

# Display the selected picture info
if st.session_state.selected_image is not None:
    st.sidebar.write(f"Selected: Picture {st.session_state.selected_image + 1}")
else:
    st.sidebar.write("No picture selected.")

# Chatbot interface with conversation history
st.subheader("Chat")
chat_placeholder = st.empty()

# Display previous conversation in chat-like format
if st.session_state.conversation:
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            st.markdown(
                f'<div style="text-align: left; background-color: #DCF8C6; padding: 8px; border-radius: 10px; max-width: 70%;">{msg["content"]}</div>',
                unsafe_allow_html=True)
        elif msg["role"] == "bot":
            st.markdown(
                f'<div style="text-align: right; background-color: #E6E6E6; padding: 8px; border-radius: 10px; max-width: 70%;">{msg["content"]}</div>',
                unsafe_allow_html=True)

# Input box for user query (styled like a chatbot input box)
user_query = st.text_input("Type your message...", key="user_input")

# Handle the chat submission
if st.button("Send"):
    if st.session_state.selected_image is not None and user_query:
        # Add user message to conversation history
        st.session_state.conversation.append({"role": "user", "content": user_query})

        # Generate a bot response (simulating with the image and query)
        response = process_query(images[st.session_state.selected_image], user_query)

        # Add bot response to conversation history
        st.session_state.conversation.append({"role": "bot", "content": response})

        # Clear the input field after sending the message
        st.session_state.user_input = ""

    else:
        st.warning("Please select a picture before sending a query.")
