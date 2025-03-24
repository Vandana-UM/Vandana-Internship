import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch

# Authenticate with Hugging Face (login with your token)
login(token="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")  # Replace with your Hugging Face token

# Load the Llama 3.1-8B-Instruct model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a LinkedIn post based on the prompt
def generate_post(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(inputs['input_ids'], max_length=300, num_return_sequences=1, no_repeat_ngram_size=2)
    post = tokenizer.decode(output[0], skip_special_tokens=True)
    return post

# Streamlit app
st.title("LinkedIn Post Generator using Llama Model (meta-llama/Llama-3.2-1B-Instruct)")
st.markdown("Generate a professional LinkedIn post based on a topic or theme.")

# Input field for user prompt
user_prompt = st.text_input("Enter a topic for your LinkedIn post:")

if user_prompt:
    st.write("Generating your LinkedIn post...")
    post = generate_post(user_prompt)
    st.subheader("Generated LinkedIn Post:")
    st.write(post)

    # Add a button to copy the generated post to the clipboard (optional)
    st.text_area("Copy your generated LinkedIn post here:", value=post, height=150)
    st.write("You can now copy and paste this post to your LinkedIn profile!")
