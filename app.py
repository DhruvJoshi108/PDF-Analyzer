# =========================================================
# PDF ANALYZER BOT
# FAST + SMART PDF CHATBOT
# =========================================================

# =========================================================
# INSTALL MODULES
# =========================================================

# RUN THESE COMMANDS IN TERMINAL:

# pip install streamlit
# pip install sentence-transformers
# pip install scikit-learn
# pip install PyPDF2
# pip install nltk
# pip install torch

# =========================================================
# IMPORTS
# =========================================================

import streamlit as st
import nltk
import numpy as np
import random

from PyPDF2 import PdfReader

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# DOWNLOAD NLTK
# =========================================================

nltk.download("punkt", quiet=True)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PDF Analyzer Bot",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #020617;
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 55px;
        font-weight: bold;
        color: white;
        margin-top: 10px;
    }

    .sub-title {
        text-align: center;
        color: #cbd5e1;
        font-size: 20px;
        margin-bottom: 30px;
    }

    .stButton button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        border: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🤖 PDF Analyzer Bot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Upload any PDF and ask questions instantly.</div>',
    unsafe_allow_html=True
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model

model = load_model()

# =========================================================
# EXTRACT PDF TEXT
# =========================================================

def extract_pdf_text(pdf_file):

    text = ""

    try:

        reader = PdfReader(pdf_file)

        for page in reader.pages:

            content = page.extract_text()

            if content:

                text += content + " "

    except Exception as e:

        st.error(f"PDF Error: {e}")

    return text

# =========================================================
# SPLIT TEXT INTO CHUNKS
# =========================================================

def split_into_chunks(text, chunk_size=150):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks

# =========================================================
# CREATE EMBEDDINGS
# =========================================================

@st.cache_data
def create_embeddings(chunks):

    embeddings = model.encode(
        chunks,
        show_progress_bar=False
    )

    return embeddings

# =========================================================
# GET BEST ANSWER
# =========================================================

def get_best_answer(question):

    try:

        question = question.lower().strip()

        question_embedding = model.encode(
            [question]
        )

        similarities = cosine_similarity(
            question_embedding,
            chunk_embeddings
        )[0]

        best_index = np.argmax(similarities)

        best_score = similarities[best_index]

        answer = chunks[best_index]

        if best_score < 0.18:

            return (
                f"""

❌ No strong answer found.

📊 Similarity Score: {best_score:.2f}

Closest Match:

{answer[:1200]}

"""
            )

        return (
            f"""

✅ Similarity Score: {best_score:.2f}

📘 Answer:

{answer[:1500]}

"""
        )

    except Exception as e:

        return f"Error: {e}"

# =========================================================
# GENERATE QUIZ QUESTION
# =========================================================

def generate_question():

    try:

        random_chunk = random.choice(chunks)

        sentences = random_chunk.split(".")

        valid_sentences = []

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) > 40:

                valid_sentences.append(sentence)

        if len(valid_sentences) == 0:

            return "No question generated."

        selected = random.choice(valid_sentences)

        words = selected.split()

        if len(words) > 15:

            selected = " ".join(words[:15])

        return (
            f"""

🧠 Quiz Question

Explain:

"{selected}...?"

"""
        )

    except Exception as e:

        return f"Error: {e}"

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload PDF File",
    type=["pdf"]
)

# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    with st.spinner("⚡ Processing PDF..."):

        pdf_text = extract_pdf_text(
            uploaded_file
        )

        if pdf_text.strip() == "":

            st.error(
                "No readable text found."
            )

        else:

            # =================================================
            # CREATE CHUNKS
            # =================================================

            chunks = split_into_chunks(
                pdf_text,
                chunk_size=150
            )

            # =================================================
            # CREATE EMBEDDINGS
            # =================================================

            chunk_embeddings = create_embeddings(
                chunks
            )

            # =================================================
            # SUCCESS MESSAGE
            # =================================================

            st.success(
                "✅ PDF processed successfully!"
            )

            st.write(
                f"📚 Total Chunks Created: {len(chunks)}"
            )

            # =================================================
            # QUIZ BUTTON
            # =================================================

            st.subheader("📘 PDF Quiz Mode")

            if st.button(
                "Generate Question From PDF"
            ):

                generated_question = generate_question()

                st.info(generated_question)

            # =================================================
            # CHAT SECTION
            # =================================================

            user_question = st.chat_input(
                "Ask anything from the PDF..."
            )

            if user_question:

                st.chat_message("user").write(
                    user_question
                )

                answer = get_best_answer(
                    user_question
                )

                st.chat_message("assistant").write(
                    answer
                )

else:

    st.info(
        "📄 Upload a PDF file to begin."
    )