
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="WriteWise — Next-Word Assistant")

@st.cache_resource
def load_model(model_dir="writewise/models/writewise-gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model

tokenizer, model = load_model()

def top_k_next_tokens(prompt, top_k=5):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k:v.to("cuda") for k,v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    last_logits = logits[0, -1, :]
    probs = torch.nn.functional.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, top_k)
    tokens = topk.indices.tolist()
    return [(tokenizer.decode([t]).strip(), float(s)) for t,s in zip(tokens, topk.values)]

def generate_text(prompt, max_new_tokens=50, temperature=0.8):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k:v.to("cuda") for k,v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_k=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

st.title("WriteWise — Next-Word Prediction & Text Generator")
prompt = st.text_area("Enter your prompt:", "The future of AI is")

col1, col2 = st.columns(2)
with col1:
    if st.button("Next-Word Suggestions"):
        results = top_k_next_tokens(prompt, 5)
        for w,p in results:
            st.write(f"**{w}**  ({p:.4f})")
with col2:
    if st.button("Generate Text"):
        out = generate_text(prompt, 60, 0.8)
        st.write(out)
