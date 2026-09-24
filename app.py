import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Environment Variables
# -----------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API Key not found. Please add OPENAI_API_KEY to your .env file.")
    st.stop()


# -----------------------------
# Structured Output
# -----------------------------

class ResumeEvaluation(BaseModel):

    match_score: int = Field(
        description="Match score between 0 and 100"
    )

    candidate_summary: str

    matching_skills: List[str]

    missing_skills: List[str]

    strengths: List[str]

    weaknesses: List[str]

    hiring_recommendation: str

    justification: str


# -----------------------------
# Streamlit Page Setup
# -----------------------------

st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="📄"
)

st.title("📄 AI Resume Screening Assistant")

st.write(
    "AI Resume Screening using LangChain, RAG, FAISS and GPT-4o-mini"
)


# -----------------------------
# Job Description Form (with Button)
# -----------------------------

with st.form("jd_form"):
    job_description = st.text_area(
        "Enter Job Description",
        height=250
    )
    # Control+Enter ki jagah dedicated Submit button
    jd_submitted = st.form_submit_button("Save Job Description")

if jd_submitted:
    if job_description.strip():
        st.success("Job Description saved successfully!")
    else:
        st.warning("Please enter a Job Description.")


# -----------------------------
# Upload Resumes
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload Resume PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)


# -----------------------------
# Evaluate Button
# -----------------------------

if st.button("Evaluate Resumes", type="primary"):

    if not job_description.strip():
        st.error("Please enter a Job Description.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()


    # -----------------------------
    # Models
    # -----------------------------

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    structured_llm = llm.with_structured_output(
        ResumeEvaluation
    )


    # -----------------------------
    # Prompt
    # -----------------------------

    prompt = ChatPromptTemplate.from_template("""

You are an AI Resume Screening Assistant.

IMPORTANT RULES:

1. Use ONLY information from the uploaded resume.
2. Do not use outside information.
3. Do not invent candidate information.
4. Compare the resume with the Job Description.
5. If a skill is not found in the resume, put it in Missing Skills.
6. The justification must be based only on resume evidence.
7. Match Score must be between 0 and 100.
8. If information is not available, write "Not found in resume".

JOB DESCRIPTION:

{job_description}


RESUME CONTEXT:

{context}

""")


    chain = prompt | structured_llm


    # -----------------------------
    # Text Splitter
    # -----------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )


    all_results = []


    # -----------------------------
    # Process Each Resume
    # -----------------------------

    for uploaded_file in uploaded_files:

        st.divider()

        st.header(
            "Candidate: " + uploaded_file.name
        )


        # Save uploaded PDF temporarily

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:

            temp.write(
                uploaded_file.read()
            )

            pdf_path = temp.name


        # -----------------------------
        # PDF Loader
        # -----------------------------

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()


        # -----------------------------
        # Split Text
        # -----------------------------

        chunks = splitter.split_documents(
            documents
        )


        # -----------------------------
        # FAISS Vector Database
        # -----------------------------

        vector_db = FAISS.from_documents(
            chunks,
            embeddings
        )


        # -----------------------------
        # Retriever
        # -----------------------------

        retriever = vector_db.as_retriever(
            search_kwargs={"k": 6}
        )


        # -----------------------------
        # Retrieve Relevant Information
        # -----------------------------

        retrieved_docs = retriever.invoke(
            job_description
        )


        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )


        # -----------------------------
        # Generate Evaluation
        # -----------------------------

        result = chain.invoke({

            "job_description": job_description,

            "context": context

        })


        all_results.append(
            (uploaded_file.name, result)
        )


        # Cleanup temp file
        os.unlink(pdf_path)


        # -----------------------------
        # Display Results
        # -----------------------------

        st.subheader("Match Score")

        st.metric(
            "Score",
            str(result.match_score) + "/100"
        )


        st.subheader("Candidate Summary")

        st.write(
            result.candidate_summary
        )


        st.subheader("Matching Skills")

        for skill in result.matching_skills:

            st.write(
                "✅", skill
            )


        st.subheader("Missing Skills")

        for skill in result.missing_skills:

            st.write(
                "❌", skill
            )


        st.subheader("Strengths")

        for strength in result.strengths:

            st.write(
                "✅", strength
            )


        st.subheader("Weaknesses")

        for weakness in result.weaknesses:

            st.write(
                "⚠️", weakness
            )


        st.subheader("Hiring Recommendation")

        st.write(
            result.hiring_recommendation
        )


        st.subheader("Justification")

        st.write(
            result.justification
        )


        # -----------------------------
        # Show RAG Context
        # -----------------------------

        with st.expander(
            "Show Retrieved Resume Information"
        ):

            st.write(context)


    # -----------------------------
    # Compare Multiple Candidates
    # -----------------------------

    if len(all_results) > 1:

        st.divider()

        st.header(
            "Candidate Comparison"
        )


        for name, result in all_results:

            st.write(
                "**" + name + "**"
            )

            st.write(
                "Match Score:",
                str(result.match_score) + "/100"
            )

            st.write(
                "Recommendation:",
                result.hiring_recommendation
            )


        st.info(
            "Review the candidates using their resume evidence, "
            "matching skills, missing skills, strengths, weaknesses "
            "and justification."
        )