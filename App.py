import stream as st
from transformers import pipeline
@st.cache_resource
def load_summerizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

st.title(" AI text Summarizer")
st.write("Enter a long text below, and get a concise summary!")
