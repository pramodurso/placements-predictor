import os
import sys
import json
import uvicorn
import httpx # Added from main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Added Field from main.py
import shutil
from typing import List, Dict, Any, Optional, Literal # Added Literal from main.py
import time
from getpass import getpass
import threading

# --- Colab/Ngrok specific imports ---
from pyngrok import ngrok, conf

# --- Import your AI/ML classes ---
# (These files must be in the same directory)
try:
    # This is your existing file
    from job_recommender import JobRecommender

    # We are bringing the functions from Resume_OCR.py into this file
    import easyocr
    import numpy as np
    from pdf2image import convert_from_path
    import google.generativeai as genai
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure 'job_recommender.py' is uploaded to your Colab session.")
    print("And that you have run: !pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

try:
    # This file contains your CommunicationAnalyzer class
    from communication_analyzer import CommunicationAnalyzer
    # Note: google.generativeai as genai is already imported above
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure 'communication_analyzer.py' is uploaded to your Colab session.")
    print("And that you have run: !pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


#
# --- 1. API Key and Model Setup ---
#
# --- Get API Keys from user input ---
GOOGLE_API_KEY = getpass("Please paste your Gemini API Key (from AI Studio): ")
if not GOOGLE_API_KEY:
    print("Gemini API key is required. Exiting.")
    sys.exit(1)

NGROK_AUTHTOKEN = getpass("Please paste your Ngrok Authtoken (from [ngrok.com/dashboard](https://ngrok.com/dashboard)): ")
if not NGROK_AUTHTOKEN:
    print("Ngrok authtoken is required. Exiting.")
    sys.exit(1)

# Configure Gemini
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    sys.exit(1)

# Configure Ngrok
try:
    conf.get_default().auth_token = NGROK_AUTHTOKEN
except Exception as e:
    print(f"Error configuring Ngrok: {e}")
    sys.exit(1)

# --- Pre-load AI Models (Global State) ---
print("Server is starting... Loading AI models. This may take a moment.")
try:
    # Load Job Recommender (loads sentence-transformer model)
    recommender = JobRecommender()

    # Load EasyOCR (downloads model)
    print("Initializing EasyOCR... This may take a moment (one-time setup).")
    ocr_reader = easyocr.Reader(['en'])

    print("AI models loaded successfully - Job Recommender")

except Exception as e:
    print(f"Fatal error during model loading: {e}")
    print("This could be due to the numpy 1.x vs 2.x issue or a network problem.")
    print("Please ensure you have run: !pip install -r requirements.txt")
    print("Also ensure your 'local-model' folder is set up correctly in 'job_recommender.py' if you are offline.")
    sys.exit(1)

try:
    # Load Communication Analyzer (loads Whisper, Parselmouth, and Gemini)
    analyzer = CommunicationAnalyzer(gemini_api_key=GOOGLE_API_KEY)

    print("AI models loaded successfully - Voice Analyzer")

except Exception as e:
    print(f"Fatal error during model loading: {e}")
    print("This could be due to a network problem or a dependency issue (like numpy).")
    print("Please ensure you have run: !pip install -r requirements.txt")
    sys.exit(1)


#
# --- 2. Define AI Pipeline Functions (from Resume_OCR.py) ---
#

def detect_document_text(path: str, reader: easyocr.Reader) -> str:
    """Uses EasyOCR to extract text from a PDF path."""
    try:
        images = convert_from_path(path)
        if not images:
            raise Exception("PDF is empty or could not be read.")
    except Exception as e:
        print(f"Error converting PDF. Is 'poppler' installed? Error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF conversion failed. Is poppler installed? Error: {e}")

    full_text = []
    for i, img in enumerate(images):
        img_np = np.array(img)
        result = reader.readtext(img_np, detail=0, paragraph=True)
        full_text.append("\n".join(result))

    return "\n\n--- END OF PAGE ---\n\n".join(full_text)

def parse_resume_to_json(resume_text: str) -> dict:
    """Uses Gemini API to parse raw text into a structured JSON."""

    # --- UPDATED SCHEMA ---
    # This schema includes all the fields your new frontend expects.
    json_schema = {
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING", "description": "The full name of the candidate."},
            "contact_info": {
                "type": "OBJECT",
                "properties": {
                    "email": {"type": "STRING"},
                    "phone": {"type": "STRING"},
                    "linkedin": {"type": "STRING", "description": "Full LinkedIn URL, if available."}
                }
            },
            "professional_summary": {"type": "STRING", "description": "A 2-4 sentence summary from the top of the resume."},
            "education": {
                "type": "OBJECT",
                "properties": {
                    "place": {"type": "STRING"},
                    "degree": {"type": "STRING"},
                    "field": {"type": "STRING"},
                    "year": {"type": "STRING"},
                    "cgpa": {"type": "STRING"},
                },
            },
            "skills": {
                "type": "OBJECT",
                "properties": {
                    "Languages": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "Backend": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "Databases": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "DevOps & Tools": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "Machine Learning": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "Generative AI": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "Other": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
            },
            "projects": {
                "type": "ARRAY",
                "description": "An array of project objects.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING", "description": "Title of the project"},
                        "skills": {"type": "STRING", "description": "Comma-separated string of skills"},
                        "desc": {"type": "STRING", "description": "Description of the project"}
                    },
                    "required": ["title", "skills", "desc"]
                }
            },
            "achievements_and_awards": {
                "type": "ARRAY",
                "description": "A list of achievements, hackathon wins, or awards.",
                "items": {"type": "STRING"}
            },
            "positions_of_responsibility": {
                "type": "ARRAY",
                "description": "A list of positions held, like 'Club Lead' or 'Coordinator'.",
                "items": {"type": "STRING"}
            }
        },
    }

    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash-preview-09-2025',
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=json_schema
            )
        )
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini model init failed: {e}")

    prompt = f"""
    You are an expert resume parser. Read the following resume text, which was extracted by OCR,
    and parse it *exactly* into the provided JSON schema.

    - Extract all relevant fields.
    - If a field is not present, return a null or empty value for it.
    - Categorize skills into the correct categories. If a skill doesn't fit, place it in "Other".
    - Extract all projects into a JSON array.
    - Extract all achievements and positions of responsibility.
    - The text "--- END OF PAGE ---" indicates a page break in the original PDF.

    RESUME TEXT:
    ---
    {resume_text}
    ---
    """

    max_retries = 3
    response = None # Initialize response to avoid reference before assignment
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            parsed_json = json.loads(response.parts[0].text)
            return parsed_json
        except Exception as e:
            print(f"Error during Gemini API call (Attempt {attempt + 1}/{max_retries}): {e}")
            if response and hasattr(response, 'parts') and response.parts:
                print("Full Gemini Response Text:", response.parts[0].text)
            time.sleep(1)

    print("Failed to parse resume after all retries.")
    raise HTTPException(status_code=500, detail="Gemini parsing failed after multiple retries.")


def convert_gemini_to_recommender_format(gemini_data: dict) -> dict:
    """
    Converts the 'projects' array from Gemini into the
    object/dict format that the JobRecommender expects.
    """
    projects_object = {}
    if "projects" in gemini_data and isinstance(gemini_data["projects"], list):
        for i, proj in enumerate(gemini_data["projects"]):
            title = proj.get("title", f"Untitled Project {i+1}")
            if title in projects_object:
                title = f"{title}_{i}"
            projects_object[title] = {
                "skills": proj.get("skills", ""),
                "desc": proj.get("desc", "")
            }

    return {
        "education": gemini_data.get("education", {}),
        "skills": gemini_data.get("skills", {}),
        "projects": projects_object
    }

#
# --- 3. FastAPI App & Endpoints ---
#
app = FastAPI(
    title="Multi-Purpose AI API",
    description="Serves Resume Analysis, Audio Analysis, and Skills Testing."
)

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#
# --- 3A. Pydantic Models ---
#

# --- Models for Resume/Job Analysis ---
class SkillGap(BaseModel):
    basic_missing: List[str]
    intermediate_missing: List[str]
    advanced_missing: List[str]

class Recommendation(BaseModel):
    job_profile: str
    match_score: float
    skill_gap: SkillGap

class ResumeAnalysisResponse(BaseModel):
    message: str
    recommendations: List[Recommendation]
    parsed_data: Dict[str, Any]  # This will hold the full JSON from Gemini

class AudioAnalysisResponse(BaseModel):
    message: str
    report: Dict[str, Any]

# --- Models for Skills Tester (from main.py) ---
class TestConfig(BaseModel):
    """Request model for configuring the test."""
    skills: List[str] = Field(..., min_length=1)
    difficulty: Literal['Easy', 'Medium', 'Hard']
    num_questions: int = Field(5, gt=0, le=10)

class Question(BaseModel):
    """Response model for a single question."""
    id: int
    type: Literal['multiple-choice', 'open-ended', 'coding']
    question: str
    topics: List[str]
    options: Optional[List[str]] = None
    answer: str # For MC, this is the exact option string. For others, it's the model answer.


#
# --- 3B. Helper Functions for Skills Tester ---
#
def get_gemini_schema():
    """Defines the JSON schema for the LLM's response."""
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "id": {"type": "INTEGER", "description": "A unique integer ID for the question, e.g., 1"},
                "type": {
                    "type": "STRING",
                    "enum": ["multiple-choice", "open-ended", "coding"],
                    "description": "The type of question."
                },
                "question": {"type": "STRING", "description": "The text of the question."},
                "topics": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "A list of topics (from the user's input) this question covers."
                },
                "options": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "A list of 4 options. REQUIRED for 'multiple-choice', null otherwise."
                },
                "answer": {
                    "type": "STRING",
                    "description": "The correct answer. For 'multiple-choice', must be one of the options. For 'coding', provide a model solution. For 'open-ended', provide a detailed model answer."
                }
            },
            "required": ["id", "type", "question", "topics", "answer"]
        }
    }

def create_system_prompt(config: TestConfig) -> str:
    """Creates the system instruction for the LLM."""
    skill_list = ", ".join(config.skills)
    return (
        "You are an expert technical interviewer and educator. Your task is to generate a set of high-quality "
        f"{config.difficulty}-level interview questions based on a specific list of topics. "
        "Follow all instructions precisely."
    )

def create_user_prompt(config: TestConfig) -> str:
    """Creates the user-facing prompt for the LLM."""
    skill_list = ", ".join(config.skills)
    
    # Ensure at least one coding question if relevant skills are present
    coding_hint = ""
    coding_skills = ['python', 'fastapi', 'javascript', 'react', 'sql', 'html/css', 'data structures', 'algorithms']
    if any(skill.lower() in coding_skills for skill in config.skills):
        coding_hint = "You MUST include at least one 'coding' question. "

    return (
        f"Generate exactly {config.num_questions} {config.difficulty}-level technical questions based on these topics: {skill_list}. "
        f"{coding_hint}"
        "For each question, provide all fields as defined in the JSON schema. "
        "For 'multiple-choice' questions, you MUST provide exactly 4 options and the 'answer' must be one of them. "
        "For 'coding' questions, the 'question' should be a problem statement and the 'answer' should be a complete, correct code solution. "
        "For 'open-ended' questions, the 'answer' should be a comprehensive, correct explanation. "
        "Ensure the 'topics' for each question are a subset of the list I provided. "
        "Return ONLY the JSON array."
    )


#
# --- 3C. API Endpoints ---
#
@app.get("/")
def read_root():
    return {
        "resume_analyzer_status": {
            "message": "Resume Analyzer API is running. POST to /analyze_resume/ to upload a PDF."
        },
        "communication_analyzer_status": {
            "message": "Communication Analyzer API is running. POST to /analyze_audio/ to upload an audio file."
        },
        "skills_tester_status": {
            "message": "Skills Tester API is running. POST to /generate-questions to create a test."
        }
    }


@app.post("/analyze_resume/", response_model=ResumeAnalysisResponse)
async def analyze_resume(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    temp_pdf_path = f"temp_{file.filename}"
    try:
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary file: {e}")
    finally:
        file.file.close()

    try:
        # --- Run the Full AI Pipeline ---
        print(f"Processing: {temp_pdf_path}")
        raw_text = detect_document_text(temp_pdf_path, ocr_reader)
        if not raw_text:
            raise HTTPException(status_code=500, detail="OCR failed to extract any text.")

        print("Parsing text with Gemini...")
        gemini_parsed_data = parse_resume_to_json(raw_text)
        if not gemini_parsed_data:
            raise HTTPException(status_code=500, detail="Gemini failed to parse the resume text.")

        # We need this converted data just for the recommender
        structured_data_for_recommender = convert_gemini_to_recommender_format(gemini_parsed_data)

        print("Getting job recommendations...")
        recommendations = recommender.recommend_jobs(
            structured_data_for_recommender["education"],
            structured_data_for_recommender["skills"],
            structured_data_for_recommender["projects"],
            top_k=3
        )

        print("Analysis complete.")

        # --- UPDATED Return ---
        return {
            "message": "Analysis successful!",
            "recommendations": recommendations,
            "parsed_data": gemini_parsed_data  # Send the full, detailed JSON
        }

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


@app.post("/analyze_audio/", response_model=AudioAnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):

    # Check if the file is an audio file
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file (e.g., .wav, .webm, .mp4).")

    # Save the audio to a temporary file
    temp_audio_path = f"temp_{file.filename}"
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary file: {e}")
    finally:
        file.file.close()

    try:
        # --- Run the Full AI Pipeline ---
        print(f"Processing: {temp_audio_path}")

        # The analyzer class handles everything:
        report = analyzer.get_final_analysis(temp_audio_path)

        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])

        print("Analysis complete.")
        return {
            "message": "Analysis successful!",
            "report": report
        }

    except Exception as e:
        # Catch any other errors
        print(f"An error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


# --- NEW ENDPOINT (from main.py, but adapted) ---
@app.post("/generate-questions", response_model=List[Question])
async def generate_questions(config: TestConfig):
    """
    Generates a list of questions based on selected skills and difficulty.
    Uses the genai library, not httpx, to respect the Colab auth flow.
    """
    system_prompt = create_system_prompt(config)
    user_prompt = create_user_prompt(config)
    schema = get_gemini_schema()
    
    response = None # Initialize for error logging

    try:
        # 1. Initialize the model with the system prompt and JSON schema
        model = genai.GenerativeModel(
            'gemini-2.5-flash-preview-09-2025',
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.8
            )
        )
        
        # 2. Call the API asynchronously
        # Use generate_content_async since this is an async def function
        response = await model.generate_content_async(user_prompt)

        # 3. Parse the response
        json_string = response.parts[0].text
        question_data_list = json.loads(json_string)
        
        # 4. Validate with Pydantic
        validated_questions = [Question(**q) for q in question_data_list]
        return validated_questions

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if response:
             # Check for safety blocks
            if response.prompt_feedback.block_reason:
                print(f"Request blocked for safety reasons: {response.prompt_feedback.block_reason}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Request blocked by safety filter: {response.prompt_feedback.block_reason}"
                )
            # Log the raw text if parsing failed
            if hasattr(response, 'parts') and response.parts:
                print(f"Gemini response text (on error): {response.parts[0].text}")

        # Handle Pydantic validation errors specifically
        if "validation error" in str(e):
             raise HTTPException(status_code=500, detail=f"LLM data validation error: {e}")
             
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


#
# --- 4. Colab/Ngrok Specific Launch Code ---
#
def run_with_ngrok(app):
    """
    Starts the FastAPI server (uvicorn) and creates a
    public URL for it using ngrok.
    """

    # Start ngrok
    print("Starting ngrok tunnel...")
    public_url = None # Initialize public_url
    try:
        public_url = ngrok.connect(8000)
        print("---")
        print(f"✅ Your backend is running at: {public_url}")
        print("---")
        print("Your frontend application (index.html) is configured to use an ngrok URL.")
        print("If the app stops working, you may need to update the URL in index.html to match the one above.")
        print(f"Resume Endpoint: {public_url}/analyze_resume/")
        print(f"Audio Endpoint:  {public_url}/analyze_audio/")
        print(f"Skills Tester:   {public_url}/generate-questions") # --- ADDED THIS LINE ---
        print("---")

    except Exception as e:
        print(f"Error starting ngrok: {e}")
        print("This may be due to a missing or invalid Authtoken.")
        sys.exit(1)

    # Start uvicorn server in a new thread
    def run_uvicorn():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

    uvicorn_thread = threading.Thread(target=run_uvicorn)
    uvicorn_thread.start()

    # Keep the main thread alive (e.g., in Colab)
    try:
        # Keep the Colab cell running
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        if public_url:
            ngrok.disconnect(public_url)
        print("Server shut down.")

# --- Start the server ---
# This is the last line of the script.
# When you run this cell in Colab, this function will be called.
run_with_ngrok(app)