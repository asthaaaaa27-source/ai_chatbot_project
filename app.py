import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ---------------------------
# 1. PAGE CONFIGURATION
# ---------------------------
st.set_page_config(page_title="AI Emotional Chatbot", page_icon="🤖")

st.title("🤖 AI Emotional Chatbot – Mini Conversational Assistant")
st.write(
    "This chatbot uses Natural Language Processing and Emotion Detection "
    "to provide supportive and intelligent responses."
)

# ---------------------------
# 2. LOAD MODELS
# ---------------------------

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="joeddav/distilbert-base-uncased-go-emotions-student",
        return_all_scores=False,
    )

@st.cache_resource
def load_chatbot_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

emotion_classifier = load_emotion_model()
tokenizer, model = load_chatbot_model()

# ---------------------------
# 3. CHAT HISTORY
# ---------------------------

if "chat_history_ids" in st.session_state:
    chat_history_ids = st.session_state["chat_history_ids"]
else:
    chat_history_ids = None

# ---------------------------
# 4. CHAT INPUT BOX
# ---------------------------

user_input = st.text_input("Type your message here…", "")

if st.button("Send") and user_input.strip() != "":
    # Detect emotion
    emotion = emotion_classifier(user_input)[0]["label"]

    st.markdown(f"**Detected Emotion:** `{emotion}`")

    # System guidance fused with user message
    system_note = (
        f"(The user seems to feel {emotion}. "
        "Respond with clarity, empathy, and a feminist supportive tone.)\n"
    )

    full_input = system_note + "User: " + user_input

    # Prepare bot input
    input_ids = tokenizer.encode(full_input + tokenizer.eos_token, return_tensors="pt")

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
    else:
        bot_input_ids = input_ids

    # Generate reply
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=150,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.92,
        top_k=50,
    )

    # Save state
    st.session_state["chat_history_ids"] = chat_history_ids

    # Decode response
    bot_reply = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True,
    )

    # Display
    st.markdown(f"### 🤖 Chatbot:\n{bot_reply}")

# ---------------------------
# 5. FOOTER
# ---------------------------
st.write("---")
st.write("Made with ❤️ by Chhavi (Class 12)")
