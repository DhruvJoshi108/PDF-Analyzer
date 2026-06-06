import random

import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_SIMILARITY = 0.18


st.set_page_config(
    page_title="PDF Analyzer Bot",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #0b1020;
        color: #f8fafc;
    }

    .main-title {
        color: #f8fafc;
        font-size: 48px;
        font-weight: 750;
        line-height: 1.05;
        margin: 12px 0 6px;
        text-align: center;
    }

    .sub-title {
        color: #cbd5e1;
        font-size: 18px;
        margin-bottom: 28px;
        text-align: center;
    }

    .stButton > button {
        background: #2563eb;
        border: 0;
        border-radius: 8px;
        color: #ffffff;
        font-size: 16px;
        height: 46px;
        width: 100%;
    }

    .metric-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 18px 0 24px;
    }

    .metric-box {
        background: #111827;
        border: 1px solid #243244;
        border-radius: 8px;
        padding: 14px 16px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 4px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 22px;
        font-weight: 700;
    }

    @media (max-width: 720px) {
        .main-title {
            font-size: 34px;
        }

        .metric-strip {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading language model...")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def extract_pdf_text(pdf_file) -> tuple[str, int]:
    text_parts: list[str] = []
    page_count = 0

    try:
        reader = PdfReader(pdf_file)
        page_count = len(reader.pages)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception as exc:
        st.error(f"PDF error: {exc}")

    return " ".join(text_parts), page_count


def split_into_chunks(text: str, chunk_size: int = 150) -> tuple[str, ...]:
    words = text.split()
    return tuple(
        " ".join(words[index : index + chunk_size])
        for index in range(0, len(words), chunk_size)
    )


@st.cache_data(show_spinner=False)
def create_embeddings(chunks: tuple[str, ...]) -> np.ndarray:
    model = load_model()
    return model.encode(chunks, show_progress_bar=False)


def get_best_answer(
    question: str,
    chunks: tuple[str, ...],
    chunk_embeddings: np.ndarray,
) -> tuple[str, float]:
    model = load_model()
    question_embedding = model.encode([question.lower().strip()])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    if best_score < MIN_SIMILARITY:
        return (
            "I could not find a strong answer in the PDF. "
            f"The closest matching section was:\n\n{chunks[best_index][:1200]}",
            best_score,
        )

    return chunks[best_index][:1500], best_score


def generate_question(chunks: tuple[str, ...]) -> str:
    random_chunk = random.choice(chunks)
    sentences = [
        sentence.strip()
        for sentence in random_chunk.split(".")
        if len(sentence.strip()) > 40
    ]

    if not sentences:
        return "No suitable question could be generated from this PDF."

    selected = random.choice(sentences)
    words = selected.split()
    prompt = " ".join(words[:15]) if len(words) > 15 else selected
    return f'Explain this from the PDF: "{prompt}..."'


st.markdown(
    '<div class="main-title">PDF Analyzer Bot</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Upload a PDF and ask questions from its contents.</div>',
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF file to begin.")
    st.stop()


with st.spinner("Processing PDF..."):
    pdf_text, page_count = extract_pdf_text(uploaded_file)

if not pdf_text.strip():
    st.error("No readable text was found in this PDF.")
    st.stop()


chunks = split_into_chunks(pdf_text)

with st.spinner("Creating searchable embeddings..."):
    chunk_embeddings = create_embeddings(chunks)


word_count = len(pdf_text.split())

st.success("PDF processed successfully.")
st.markdown(
    f"""
    <div class="metric-strip">
        <div class="metric-box">
            <div class="metric-label">Pages</div>
            <div class="metric-value">{page_count}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Words</div>
            <div class="metric-value">{word_count}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Chunks</div>
            <div class="metric-value">{len(chunks)}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


quiz_col, chat_col = st.columns([1, 2], gap="large")

with quiz_col:
    st.subheader("Quiz Mode")
    if st.button("Generate Question"):
        st.info(generate_question(chunks))

with chat_col:
    st.subheader("Ask From The PDF")
    user_question = st.chat_input("Ask anything from the PDF...")

    if user_question:
        st.chat_message("user").write(user_question)
        answer, score = get_best_answer(user_question, chunks, chunk_embeddings)
        st.chat_message("assistant").write(answer)
        st.caption(f"Similarity score: {score:.2f}")
