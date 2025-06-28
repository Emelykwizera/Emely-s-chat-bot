import streamlit as st
import PyPDF2
import nltk
import os

# Create a folder 'nltk_data' in your app directory
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Tell nltk to look in that folder
nltk.data.path.append(nltk_data_dir)

# Download punkt tokenizer data into that folder if missing
nltk.download("punkt", download_dir=nltk_data_dir)
 
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🧠 Download tokenizer only once


# 📄 Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# 📖 Load and process the handbook
handbook_text = extract_text_from_pdf("Kepler_college_Student_Handbook.pdf")
sentences = sent_tokenize(handbook_text)

# 🔍 Convert sentences to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)


# 🎛️ Streamlit UI
st.title("🎓 Emely's Chatbot")
st.write("Ask me anything about the Kepler Student handbook!")

# 📥 Input from the user
user_question = st.text_input("Your question:")

# 🧠 If there's a question, process it
if user_question:
    question_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(question_vector, tfidf_matrix)
    best_match_index = similarities.argmax()
    answer = sentences[best_match_index]

    # 📤 Show result
    st.subheader("Answer:")
    st.write(answer)
