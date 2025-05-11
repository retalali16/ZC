import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

# Load BERT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

    def forward(self, input1, input2):
        return torch.nn.functional.cosine_similarity(input1, input2, dim=1)

siamese_network = SiameseNetwork()

# Embedding function
def get_bert_embeddings(text):
    if isinstance(text, str):
        text = [text]
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Compare questions
def compare_questions(q1, q2):
    emb1 = get_bert_embeddings(q1)
    emb2 = get_bert_embeddings(q2)
    similarity = siamese_network(emb1, emb2)
    return similarity.item()

# Predefined Q&A
predefined_answers = {
    "What are the admission requirements for Zewail City?": "Applicants must have a high school diploma (Thanaweya Amma or equivalent)...",
    "How can I apply to Zewail City?": "You can apply through the official Zewail City website...",
    "What are the available programs at Zewail City?": "Zewail City offers programs in Engineering, CS, Nanotech...",
    "Is there a scholarship program at Zewail City?": "Yes, Zewail City offers merit-based and need-based scholarships...",
    "What is the tuition fee for undergraduate programs?": "Tuition fees vary by program. Check the website for details.",
    "What is the deadline for admission applications?": "Deadlines are announced on the website. Apply early!",
    "Do international students qualify for admission?": "Yes, international students can apply if they meet the criteria.",
    "What entrance exams are required for admission?": "Entrance exams in math, physics, etc., may be required.",
    "How do I contact the admission office?": "Email admissions@zewailcity.edu.eg or call the numbers listed online.",
    "Is there student accommodation available?": "Yes, there is on-campus housing subject to availability."
}

# Streamlit UI
st.title("🎓 Zewail City Chatbot")
st.write("Ask me anything about Zewail City!")

user_input = st.text_input("Your question:")

if user_input:
    best_answer = "Sorry, I didn’t understand that."
    highest_score = -1

    for question, answer in predefined_answers.items():
        score = compare_questions(user_input, question)
        if score > highest_score:
            highest_score = score
            best_answer = answer

    st.subheader("💬 Answer")
    st.write(best_answer)
