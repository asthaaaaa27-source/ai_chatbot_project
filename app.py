import streamlit as st
import requests

# ------------------ PAGE SETTINGS ------------------
st.set_page_config(page_title="Chhavi's AI Chatbot", page_icon="🤖", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            text-align: center;
            font-weight: bold;
            color: #6C63FF;
        }
        .chat-bubble-user {
            background-color: #6A63FF;
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 8px;
            max-width: 70%;
            float: right;
        }
        .chat-bubble-bot {
            background-color: #6A63FF;
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 8px;
            max-width: 70%;
            float: left;
        }
        .chat-box {
            height: 400px;
            overflow-y: auto;
            padding: 10px;
            border: 2px solid #DDD;
            border-radius: 10px;
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ HUGGING FACE API SETTINGS ------------------
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
API_KEY = "PUT_YOUR_API_KEY_HERE"
headers = {"Authorization": f"Bearer {API_KEY}"}

def generate_response(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        return response.json()[0]["generated_text"]
    except:
        return "Sorry, I couldn't generate a response."

# ------------------ SESSION STATE FOR CHAT HISTORY ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ UI TITLE ------------------
st.markdown('<div class="title">💜 Intelligent AI Assistant 💬</div>', unsafe_allow_html=True)
st.write("")

# ------------------ CHAT DISPLAY ------------------
st.write("### Chat History")
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-box" id="chat-box">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div><br>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-bot'>{msg['content']}</div><br>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ INPUT AREA ------------------
user_input = st.text_input("Type your message here, Chhavi:")

if st.button("Send"):
    if user_input.strip() != "":
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate bot reply
        bot_response = generate_response(user_input)
        st.session_state.messages.append({"role": "bot", "content": bot_response})

        st.rerun()




