import os
import psycopg
from fastapi import FastAPI, Depends, HTTPException, status,Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dotenv import load_dotenv
from urllib.parse import quote_plus 
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from typing import Annotated

# 1. Load variables from .env file
load_dotenv()

# 2. Create FastAPI app
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str | None = None

class StudentProfileCreate(BaseModel):
    full_name: str
    course: str
    tenth_percentage: float
    twelfth_percentage: float
    current_year: int
    cgpa: float
    skills: list[str]  # A list of strings
    projects: str | None = None
    achievements: str | None = None
    linkedin_url: str | None = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 3. Get credentials from environment
db_user = os.getenv('DB_USER')
db_pass_raw = os.getenv('DB_PASS')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

# 4. URL-encode the password
db_pass_encoded = quote_plus(db_pass_raw)

# 5. Build the Database Connection String
DB_URL = (
    f"postgresql://{db_user}:{db_pass_encoded}"
    f"@{db_host}:{db_port}/{db_name}"
)

print(f"DEBUG: Connecting with URL: {DB_URL}")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """Creates a new JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Default to 15 minutes if no time is given
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, credentials_exception):
    """Checks a JWT token for validity."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
        return token_data
    except JWTError:
        raise credentials_exception
    

# 6. Create the database dependency
def get_db_connection():
    try:
        with psycopg.connect(DB_URL) as conn:
            yield conn
    # --- CHANGE 1: Be more specific about the error ---
    # We only want to catch database-specific errors here.
    except psycopg.OperationalError as e:
        print(f"ERROR: Could not connect to database: {e}")
        raise HTTPException(status_code=500, detail="Database connection error.")
    

# --- NEW: Dependency to get the current user from a token ---
def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: psycopg.Connection = Depends(get_db_connection)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 1. Verify the token is valid and get the email from it
    token_data = verify_token(token, credentials_exception)

    # 2. Get the user's full details from the database
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT id, email, role FROM users WHERE email = %s",
                (token_data.email,)
            )
            user_record = cur.fetchone()
            if user_record is None:
                raise credentials_exception

            # Return the user's info as a simple dictionary
            return {
                "id": user_record[0],
                "email": user_record[1],
                "role": user_record[2]
            }
    except Exception:
        raise credentials_exception
    

# 7. Create our test endpoint
@app.get("/")
def check_db_connection(
    db: psycopg.Connection = Depends(get_db_connection)
):
    try:
        with db.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
            
            if result == (1,):
                return {
                    "status": "success", 
                    "message": "FastAPI is running and connected to PostgreSQL!"
                }
            else:
                raise HTTPException(status_code=500, detail="Database connection test failed.")
    
    except Exception as e:
        print(f"Database query error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Database query error: {e}"
        )

# --- NEW: Endpoint for Step 1 - User Registration ---
@app.post("/register")
def register_user(
    user_data: UserCreate,
    db: psycopg.Connection = Depends(get_db_connection)
):
    plain_password = user_data.password

    if user_data.role not in ['student', 'officer']:
        raise HTTPException(
            status_code=400, 
            detail="Invalid role. Must be 'student' or 'officer'."
        )

    # 5. Try to hash password AND insert the new user
    try:
        # --- CHANGE 2: Move hashing INSIDE the try block ---
        # Now, if hashing fails, our 'except Exception as e' will catch it.
        hashed_password = pwd_context.hash(plain_password)

        with db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (email, password_hash, role)
                VALUES (%s, %s, %s)
                """,
                (user_data.email, hashed_password, user_data.role)
            )
            db.commit()
            
    except psycopg.IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=400, 
            detail="Email already registered."
        )
    except Exception as e:
        # This will now catch the bcrypt error and show it to us
        db.rollback()
        print(f"ERROR during registration: {e}") # <--- This will show the REAL error
        raise HTTPException(
            status_code=500,
            # We also send the real error back to the browser
            detail=f"An error occurred: {e}"
        )

    # 10. If everything worked, return a success message
    return {
        "status": "success",
        "message": f"User '{user_data.email}' created as a '{user_data.role}'."
    }




# --- REVISED AGAIN: Endpoint for Step 1 - User Login ---
# This version uses the standard OAuth2PasswordRequestForm
@app.post("/login", response_model=Token)
def login_user(
    # This special class is built to handle the "Authorize" popup
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: psycopg.Connection = Depends(get_db_connection)
):
    # The form gives us "username", so we assign it to an 'email' variable
    email = form_data.username
    password = form_data.password

    # 1. Try to find the user in the database
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT email, password_hash, role FROM users WHERE email = %s",
                (email,)  # Use the email variable
            )
            user_record = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {e}")

    # 2. Check if user exists and password is correct
    if not user_record or not pwd_context.verify(password, user_record[1]): # Use the password variable
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. If they are real, make their "wristband" (token)
    user_email, user_hash, user_role = user_record
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_email, "role": user_role}, 
        expires_delta=access_token_expires
    )

    # 4. Hand them the token
    return {"access_token": access_token, "token_type": "bearer"}



# --- NEW: Endpoint for Step 2a - Create/Update Student Profile ---
@app.post("/profile/update")
def update_student_profile(
    profile_data: StudentProfileCreate,
    current_user: dict = Depends(get_current_user),
    db: psycopg.Connection = Depends(get_db_connection)
):
    # 1. Check if the user is a student
    if current_user["role"] != 'student':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can create a profile."
        )

    user_id = current_user["id"]

    # 2. SQL to insert or update the profile
    # "ON CONFLICT" is a powerful PostgreSQL command.
    # It means: "Try to INSERT. If a profile for this user_id
    # already exists, UPDATE it instead."
    sql_query = """
    INSERT INTO student_profiles (
        user_id, full_name, course, tenth_percentage, twelfth_percentage,
        current_year, cgpa, skills, projects, achievements, linkedin_url,
        last_updated
    )
    VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
    )
    ON CONFLICT (user_id) DO UPDATE SET
        full_name = EXCLUDED.full_name,
        course = EXCLUDED.course,
        tenth_percentage = EXCLUDED.tenth_percentage,
        twelfth_percentage = EXCLUDED.twelfth_percentage,
        current_year = EXCLUDED.current_year,
        cgpa = EXCLUDED.cgpa,
        skills = EXCLUDED.skills,
        projects = EXCLUDED.projects,
        achievements = EXCLUDED.achievements,
        linkedin_url = EXCLUDED.linkedin_url,
        last_updated = NOW();
    """

    # 3. Try to execute the query
    try:
        with db.cursor() as cur:
            cur.execute(sql_query, (
                user_id,
                profile_data.full_name,
                profile_data.course,
                profile_data.tenth_percentage,
                profile_data.twelfth_percentage,
                profile_data.current_year,
                profile_data.cgpa,
                profile_data.skills,
                profile_data.projects,
                profile_data.achievements,
                profile_data.linkedin_url
            ))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error updating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}"
        )

    return {
        "status": "success",
        "message": "Profile updated successfully.",
        "user_id": user_id
    }