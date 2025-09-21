from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import docx
import openai
import json
app = Flask(__name__)
CORS(app) # <--- And make sure this line is right after you create the app


# --- Configuration ---
# Set your OpenAI API key as an environment variable for security
# In your terminal:
# For Windows: set OPENAI_API_KEY=your_key_here
# For macOS/Linux: export OPENAI_API_KEY=your_key_here
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)  # This allows your frontend at 127.0.0.1 to talk to this backend

# --- Helper Function to Extract Text ---
def extract_text_from_file(file):
    """Extracts text from an uploaded file (PDF or DOCX)."""
    filename = file.filename
    try:
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file.stream)
            text = "".join(page.extract_text() for page in pdf_reader.pages)
            return text
        elif filename.endswith('.docx'):
            doc = docx.Document(file.stream)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        else:
            return None
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return ""

# --- Core AI Analysis Function ---
def get_consistent_analysis(jd_text, resume_text, resume_filename):
    """
    Analyzes resume against job description using an LLM with temperature=0.
    """
    # A more detailed prompt to get the exact JSON structure your frontend needs
    prompt = f"""
    Analyze the provided resume against the job description. Based on the analysis, return a single JSON object with the following exact keys:
    - "score": An integer between 0 and 100 representing the percentage match.
    - "verdict": A single word string: "High", "Medium", or "Low".
    - "candidateName": A string with the candidate's name. If you can't find it, use the filename "{resume_filename}".
    - "matchedSkills": An array of strings listing the key skills from the JD found in the resume.
    - "missingSkills": An array of strings listing important skills from the JD NOT found in the resume.
    - "suggestions": An array of 2-3 brief, actionable suggestions for the candidate.

    Do not include any text outside of the single, valid JSON object.

    --- Job Description ---
    {jd_text}

    --- Candidate's Resume ---
    {resume_text}

    --- JSON Output ---
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",  # This model is good at returning JSON
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            # THIS IS THE FIX: temperature=0 makes the output deterministic
            temperature=0,
            max_tokens=1000
        )
        # The API response content is a JSON string, so we parse it
        result_json = json.loads(response.choices[0].message.content)
        return result_json
    except Exception as e:
        print(f"An error occurred during OpenAI API call: {e}")
        return {"error": "Failed to get analysis from AI model."}

# --- API Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze_resume_endpoint():
    if 'jobDescription' not in request.files or 'resume' not in request.files:
        return jsonify({"error": "Missing job description or resume file"}), 400

    jd_file = request.files['jobDescription']
    resume_file = request.files['resume']

    jd_text = extract_text_from_file(jd_file)
    resume_text = extract_text_from_file(resume_file)

    if not jd_text or not resume_text:
        return jsonify({"error": "Could not read text from one or both files."}), 400

    # Get the consistent analysis from the model
    analysis_result = get_consistent_analysis(jd_text, resume_text, resume_file.filename)

    if "error" in analysis_result:
        return jsonify(analysis_result), 500

    return jsonify(analysis_result), 200

# --- Run the App ---
if __name__ == '__main__':
    app.run(port=5000, debug=True)