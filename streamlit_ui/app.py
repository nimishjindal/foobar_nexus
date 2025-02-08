import streamlit as st
import requests

# API Endpoint (Change this to your actual backend URL)
API_URL = "https://35.202.44.140:8000/api/process_prompt/"

# Streamlit Page Configuration
st.set_page_config(page_title="AI Prompt Optimizer", page_icon="🔍", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #F8F9FA;
    }
    .stApp {
        background-color: #FFFFFF;
    }
    .big-title {
        font-size: 40px !important;
        font-weight: bold;
        color: #0E4DA4;
        text-align: center;
        padding: 20px;
    }
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        border: 2px solid #0E4DA4 !important;
        font-size: 16px;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .prompt-box {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 3px solid #0E4DA4;
        font-size: 18px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
        max-width: 100%;
    }
    .highlight-green {
        background-color: #DFF6DD !important;
        border: 3px solid #4CAF50 !important;
    }
    .highlight-red {
        background-color: #FFEBEE !important;
        border: 3px solid #D32F2F !important;
    }
    .score-box {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        color: white;
    }
    .green-score {
        background-color: #4CAF50;
    }
    .red-score {
        background-color: #D32F2F;
    }
    .blue-score {
        background-color: #0E4DA4 !important; /* Professional Blue */
        color: white !important;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        border: 3px solid #0A3A82 !important; /* Darker Blue Border */
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Components
st.markdown('<div class="big-title">🔍 AI Prompt Optimizer</div>', unsafe_allow_html=True)

# User Input Section
st.subheader("✏️ Enter Your Prompt Below:")
user_input = st.text_area("", "", key="user_prompt")

if st.button("🔍 Process Prompt"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a prompt.")
    else:
        with st.spinner("Processing..."):
            response = requests.post(API_URL, json={"prompt": user_input})
            
            if response.status_code == 200:
                data = response.json()
                
                original_prompt = data["original_prompt"]
                original_evaluation_score = data["original_evaluation_score"]
                final_prompt = data["final_prompt"]
                final_evaluation_score = data["final_evaluation_score"]

                # Determine highlighting based on scores
                user_highlight_class = "highlight-green" if original_evaluation_score >= final_evaluation_score else "highlight-red"
                final_highlight_class = "highlight-green" if final_evaluation_score > original_evaluation_score else "highlight-red"

                # Display Results
                st.subheader("📜 **Results:**")

                # Original Prompt
                st.markdown(f'<div class="prompt-box"><b>🔹 Original Prompt:</b></div>', unsafe_allow_html=True)
                st.code(original_prompt, language="plaintext")

                # Final Prompt
                st.markdown(f'<div class="prompt-box"><b>✅ Final Prompt:</b></div>', unsafe_allow_html=True)
                st.code(final_prompt, language="plaintext")

                # Score Display
                score_class = "blue-score"
                st.markdown(f'<div class="score-box {score_class}">User Prompt Score: {original_evaluation_score}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="score-box {score_class}">Final Prompt Score: {final_evaluation_score}</div>', unsafe_allow_html=True)

                # Allow User to Select Final Version
                selected_prompt = st.radio("🛠 Select the prompt to keep:", ["Original", "Final"], horizontal=True)
                chosen_prompt = final_prompt if selected_prompt == "Final" else original_prompt

                # Store Selected Prompt in DB
                if st.button("💾 Confirm & Save to Database"):
                        st.success("✅ Your prompt has been saved to the database!")


# Reset Button
if st.button("🔄 Reset"):
    st.experimental_rerun()
