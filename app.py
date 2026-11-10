
import re
import fitz
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ---------------------------------
# SKILL DATABASE
# ---------------------------------

SKILL_DATABASE = {
    "Python": ["python"],
    "Java": ["java"],
    "C": ["c programming", " c language "],
    "C++": ["c++"],
    "JavaScript": ["javascript", "js"],
    "HTML": ["html"],
    "CSS": ["css"],
    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "Artificial Intelligence": ["artificial intelligence", " ai "],
    "NLP": ["natural language processing", "nlp"],
    "Data Science": ["data science"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Flask": ["flask"],
    "Django": ["django"],
    "FastAPI": ["fastapi"],
    "React": ["react", "react.js"],
    "Node.js": ["node.js", "nodejs"],
    "Git": ["git"],
    "GitHub": ["github"],
    "Docker": ["docker"],
    "AWS": ["aws", "amazon web services"],
    "Power BI": ["power bi"],
    "Excel": ["excel", "ms excel"],
    "Communication": ["communication skills"],
    "Problem Solving": ["problem solving", "problem-solving"],
}


# ---------------------------------
# JOB DATABASE
# ---------------------------------

JOB_DATABASE = [
    {
        "role": "Python Developer",
        "skills": ["Python", "SQL", "Git", "Flask", "Django"],
        "description": "Develop Python applications, APIs, backend systems, "
                       "database integrations and software solutions."
    },
    {
        "role": "Java Developer",
        "skills": ["Java", "SQL", "MySQL", "Git"],
        "description": "Develop Java applications, object oriented programs, "
                       "backend services and database applications."
    },
    {
        "role": "Machine Learning Engineer",
        "skills": ["Python", "Machine Learning", "Scikit-learn",
                   "Pandas", "NumPy"],
        "description": "Build machine learning models, process datasets, "
                       "train predictive systems and evaluate model performance."
    },
    {
        "role": "Data Analyst",
        "skills": ["Python", "SQL", "Excel", "Power BI", "Pandas"],
        "description": "Analyze datasets, create reports, visualize data "
                       "and generate business insights."
    },
    {
        "role": "Frontend Developer",
        "skills": ["HTML", "CSS", "JavaScript", "React"],
        "description": "Build responsive websites and user interfaces "
                       "using frontend technologies."
    },
    {
        "role": "Full Stack Developer",
        "skills": ["HTML", "CSS", "JavaScript", "React",
                   "Python", "SQL", "Node.js"],
        "description": "Develop complete web applications with frontend, "
                       "backend, APIs and database integration."
    },
    {
        "role": "AI Developer",
        "skills": ["Python", "Artificial Intelligence",
                   "Machine Learning", "NLP", "TensorFlow"],
        "description": "Develop artificial intelligence applications, "
                       "intelligent systems and natural language processing solutions."
    },
    {
        "role": "Backend Developer",
        "skills": ["Python", "Java", "SQL", "MySQL",
                   "FastAPI", "Docker", "Git"],
        "description": "Build backend applications, REST APIs, "
                       "database systems and server-side services."
    }
]


# ---------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------

def extract_resume_text(pdf_file):

    document = fitz.open(
        stream=pdf_file.getvalue(),
        filetype="pdf"
    )

    pages = []

    for page in document:
        pages.append(page.get_text())

    document.close()

    return "\n".join(pages).strip()


# ---------------------------------
# SKILL EXTRACTION
# ---------------------------------

def extract_skills(text):

    normalized_text = " " + text.lower() + " "
    detected_skills = []

    for skill, aliases in SKILL_DATABASE.items():

        for alias in aliases:

            pattern = r"(?<!\w)" + re.escape(alias.strip()) + r"(?!\w)"

            if re.search(pattern, normalized_text):
                detected_skills.append(skill)
                break

    return sorted(set(detected_skills))


# ---------------------------------
# JOB MATCHING
# ---------------------------------

def match_jobs(resume_text, detected_skills):

    results = []

    for job in JOB_DATABASE:

        required_skills = job["skills"]

        matched_skills = [
            skill for skill in required_skills
            if skill in detected_skills
        ]

        missing_skills = [
            skill for skill in required_skills
            if skill not in detected_skills
        ]

        skill_score = (
            len(matched_skills) / len(required_skills)
        ) * 100

        combined_text = (
            resume_text + " " +
            " ".join(detected_skills)
        )

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        try:
            vectors = vectorizer.fit_transform([
                combined_text,
                job["description"]
            ])

            similarity = cosine_similarity(
                vectors[0:1],
                vectors[1:2]
            )[0][0] * 100

        except ValueError:
            similarity = 0

        final_score = (
            0.70 * skill_score +
            0.30 * similarity
        )

        results.append({
            "Job Role": job["role"],
            "Match Score": round(final_score, 2),
            "Skill Coverage": round(skill_score, 2),
            "Matched Skills": matched_skills,
            "Missing Skills": missing_skills
        })

    return sorted(
        results,
        key=lambda x: x["Match Score"],
        reverse=True
    )


# ---------------------------------
# RESUME IMPROVEMENT SUGGESTIONS
# ---------------------------------

def generate_suggestions(text, skills):

    suggestions = []
    lower_text = text.lower()

    if len(text.split()) < 150:
        suggestions.append(
            "Your resume appears short. Add more details about "
            "your education, skills, projects and experience."
        )

    if "objective" not in lower_text and "summary" not in lower_text:
        suggestions.append(
            "Add a professional summary or career objective."
        )

    if "project" not in lower_text:
        suggestions.append(
            "Include academic or personal projects with clear descriptions."
        )

    if "education" not in lower_text:
        suggestions.append(
            "Add an education section with your qualification and institution."
        )

    if "experience" not in lower_text and "internship" not in lower_text:
        suggestions.append(
            "If applicable, include internships, training or work experience."
        )

    if "certification" not in lower_text and "certificate" not in lower_text:
        suggestions.append(
            "Include relevant technical certifications if you have them."
        )

    if len(skills) < 5:
        suggestions.append(
            "Highlight more relevant technical skills that you genuinely possess."
        )

    if not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        suggestions.append(
            "Make sure your professional email address is included."
        )

    if not suggestions:
        suggestions.append(
            "Good work! Review your resume for measurable project results, "
            "clear formatting and role-specific keywords."
        )

    return suggestions


# ---------------------------------
# APPLICATION DESIGN
# ---------------------------------

st.title("📄 AI-Powered Resume Analyzer")
st.markdown(
    "### Analyze your resume | Discover your skills | Find suitable job roles"
)

st.write(
    "Upload your PDF resume to receive skill analysis, "
    "job matching results and improvement suggestions."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload your resume (PDF only)",
    type=["pdf"]
)


# ---------------------------------
# ANALYZE BUTTON
# ---------------------------------

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button(
        "🚀 Analyze My Resume",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing your resume..."):

            try:
                resume_text = extract_resume_text(uploaded_file)

                if not resume_text:
                    st.error(
                        "No readable text was found. "
                        "Please upload a text-based PDF instead of a scanned image."
                    )
                    st.stop()

                detected_skills = extract_skills(resume_text)

                job_results = match_jobs(
                    resume_text,
                    detected_skills
                )

                suggestions = generate_suggestions(
                    resume_text,
                    detected_skills
                )

                st.session_state["analysis"] = {
                    "text": resume_text,
                    "skills": detected_skills,
                    "jobs": job_results,
                    "suggestions": suggestions
                }

            except Exception as error:
                st.error(f"Analysis failed: {error}")


# ---------------------------------
# DISPLAY RESULTS
# ---------------------------------

if "analysis" in st.session_state:

    analysis = st.session_state["analysis"]

    st.divider()
    st.header("📊 Resume Analysis Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Skills Detected",
            len(analysis["skills"])
        )

    with col2:
        st.metric(
            "Job Roles Analyzed",
            len(analysis["jobs"])
        )

    with col3:
        st.metric(
            "Suggestions",
            len(analysis["suggestions"])
        )

    # SKILLS

    st.subheader("🧠 Technical Skills Detected")

    if analysis["skills"]:
        st.write(" | ".join(analysis["skills"]))
    else:
        st.warning("No recognized skills were detected.")

    # JOB MATCHING

    st.subheader("💼 Recommended Job Roles")

    for job in analysis["jobs"]:

        with st.expander(
            f'{job["Job Role"]} — {job["Match Score"]}% match'
        ):

            st.progress(
                int(job["Match Score"])
            )

            st.write(
                f'**Skill coverage:** {job["Skill Coverage"]}%'
            )

            st.write("**Matched Skills:**")

            if job["Matched Skills"]:
                st.write(", ".join(job["Matched Skills"]))
            else:
                st.write("No matching skills found.")

            st.write("**Skills to develop:**")

            if job["Missing Skills"]:
                st.write(", ".join(job["Missing Skills"]))
            else:
                st.write("All listed skills matched!")

    # RESULTS TABLE

    st.subheader("📈 Job Match Comparison")

    job_table = pd.DataFrame([
        {
            "Job Role": job["Job Role"],
            "Match Score (%)": job["Match Score"],
            "Skill Coverage (%)": job["Skill Coverage"]
        }
        for job in analysis["jobs"]
    ])

    st.dataframe(
        job_table,
        use_container_width=True,
        hide_index=True
    )

    # SUGGESTIONS

    st.subheader("💡 Resume Improvement Suggestions")

    for index, suggestion in enumerate(
        analysis["suggestions"],
        start=1
    ):
        st.write(f"{index}. {suggestion}")

    # EXTRACTED TEXT

    with st.expander("📃 View Extracted Resume Text"):
        st.text(analysis["text"])

    # DOWNLOAD REPORT

    report = f"""
AI RESUME ANALYZER REPORT

DETECTED SKILLS:
{", ".join(analysis["skills"])}

JOB MATCHING RESULTS:
"""

    for job in analysis["jobs"]:
        report += (
            f'\n{job["Job Role"]}: {job["Match Score"]}%'
        )

    report += "\n\nIMPROVEMENT SUGGESTIONS:\n"

    for suggestion in analysis["suggestions"]:
        report += f"\n- {suggestion}"

    st.download_button(
        "⬇️ Download Analysis Report",
        data=report,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.success("Resume analysis completed successfully!")

st.divider()

st.caption(
    "AI Resume Analyzer | Built with Python, Streamlit, "
    "PyMuPDF and Scikit-learn"
)

st.caption(
    "Note: Job-match scores are estimates from a predefined "
    "job-role database. They are not employer ATS scores."
)
