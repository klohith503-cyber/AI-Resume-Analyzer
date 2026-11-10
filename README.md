
# AI-Powered Resume Analyzer & Job Matcher

An interactive Python application that analyzes PDF resumes, extracts technical skills, matches candidates with suitable predefined job roles, and provides resume improvement suggestions.

## Features

- PDF resume upload
- Automatic resume text extraction
- Technical skill detection
- AI-assisted job-role matching using TF-IDF and cosine similarity
- Job match percentage
- Matched and missing skill identification
- Resume improvement suggestions
- Downloadable analysis report

## Technologies Used

- Python
- Streamlit
- PyMuPDF
- Scikit-learn
- Pandas
- NumPy

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Deployment

This application can be deployed using Streamlit Community Cloud.

1. Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Connect your GitHub account to Streamlit Community Cloud.
3. Select the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy the application.

## Important Notes

- The application supports text-based PDF resumes.
- Scanned image-only PDFs require OCR, which is not included.
- Job roles and required skills are predefined.
- Job matching scores are estimates, not official ATS scores.
- The application does not fetch live job vacancies.

## Author

AI Resume Analyzer Project
