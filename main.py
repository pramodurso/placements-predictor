import os
import psycopg
from fastapi import FastAPI, Depends, HTTPException, status,Form
from fastapi import File, UploadFile
import time
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dotenv import load_dotenv
from urllib.parse import quote_plus 
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from typing import Annotated, List, Optional
from fastapi.middleware.cors import CORSMiddleware

# 1. Load variables
load_dotenv()

# 2. Create App & Add CORS
app = FastAPI()
origins = [
    "http://localhost", "http://localhost:8080",
    "http://127.0.0.1", "http://127.0.0.1:8080", "null"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- Pydantic Models ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str | None = None

# --- MODIFIED: StudentProfileCreate ---
# 'email' field has been REMOVED from this model.
class StudentProfileCreate(BaseModel):
    full_name: Optional[str] = None
    course: Optional[str] = None
    tenth_percentage: Optional[float] = None
    twelfth_percentage: Optional[float] = None
    current_year: Optional[int] = None
    cgpa: Optional[float] = None
    skills: Optional[List[str]] = None
    projects: Optional[str] = None
    achievements: Optional[str] = None
    linkedin_url: Optional[str] = None

# --- Config and Helper Functions (Unchanged) ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
db_user = os.getenv('DB_USER')
db_pass_raw = os.getenv('DB_PASS')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_pass_encoded = quote_plus(db_pass_raw)
DB_URL = (
    f"postgresql://{db_user}:{db_pass_encoded}"
    f"@{db_host}:{db_port}/{db_name}"
)
print(f"DEBUG: Connecting with URL: {DB_URL}")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
        return token_data
    except JWTError:
        raise credentials_exception
    
def get_db_connection():
    try:
        with psycopg.connect(DB_URL) as conn:
            yield conn
    except psycopg.OperationalError as e:
        print(f"ERROR: Could not connect to database: {e}")
        raise HTTPException(status_code=500, detail="Database connection error.")
    
def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: psycopg.Connection = Depends(get_db_connection)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = verify_token(token, credentials_exception)
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT id, email, role FROM users WHERE email = %s",
                (token_data.email,)
            )
            user_record = cur.fetchone()
            if user_record is None:
                raise credentials_exception
            return {
                "id": user_record[0],
                "email": user_record[1],
                "role": user_record[2]
            }
    except Exception:
        raise credentials_exception

# --- Endpoints (GET /, POST /register, POST /login are unchanged) ---
@app.get("/")
def check_db_connection(db: psycopg.Connection = Depends(get_db_connection)):
    try:
        with db.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
            if result == (1,):
                return {"status": "success", "message": "FastAPI is running and connected to PostgreSQL!"}
            else:
                raise HTTPException(status_code=500, detail="Database connection test failed.")
    except Exception as e:
        print(f"Database query error: {e}")
        raise HTTPException(status_code=500, detail=f"Database query error: {e}")

@app.post("/register")
def register_user(user_data: UserCreate, db: psycopg.Connection = Depends(get_db_connection)):
    plain_password = user_data.password
    if user_data.role not in ['student', 'officer']:
        raise HTTPException(status_code=400, detail="Invalid role. Must be 'student' or 'officer'.")
    try:
        hashed_password = pwd_context.hash(plain_password)
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, password_hash, role) VALUES (%s, %s, %s)",
                (user_data.email, hashed_password, user_data.role)
            )
            db.commit()
    except psycopg.IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already registered.")
    except Exception as e:
        db.rollback()
        print(f"ERROR during registration: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    return {"status": "success", "message": f"User '{user_data.email}' created as a '{user_data.role}'."}

@app.post("/login", response_model=Token)
def login_user(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: psycopg.Connection = Depends(get_db_connection)
):
    email = form_data.username
    password = form_data.password
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT email, password_hash, role FROM users WHERE email = %s",
                (email,)
            )
            user_record = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {e}")

    if not user_record or not pwd_context.verify(password, user_record[1]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_email, user_hash, user_role = user_record
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_email, "role": user_role}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# --- MODIFIED: /profile/update ---
# This endpoint is now flexible and only updates fields that are provided
@app.post("/profile/update")
def update_student_profile(
    profile_data: StudentProfileCreate,
    current_user: dict = Depends(get_current_user),
    db: psycopg.Connection = Depends(get_db_connection)
):
    if current_user["role"] != 'student':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only students can create a profile.")
    
    user_id = current_user["id"]

    # 1. Get all fields from the Pydantic model
    #    .dict() converts it to a dictionary
    #    exclude_unset=True means it only includes fields that were actually sent
    update_data = profile_data.dict(exclude_unset=True)

    # 2. If no data was sent, just return
    if not update_data:
        return {"status": "no_changes", "message": "No data provided to update."}

    # 3. Dynamically build the SQL query
    # This creates a string like "full_name = %s, course = %s"
    set_query_parts = [f"{key} = %s" for key in update_data.keys()]
    set_query_string = ", ".join(set_query_parts)

    sql_values = list(update_data.values())
    set_query_string += ", last_updated = NOW()"
    conflict_values = list(update_data.values())
    
    # We build the query with only the keys we received
    sql_query = f"""
    INSERT INTO student_profiles (user_id, {", ".join(update_data.keys())}, last_updated)
    VALUES (%s, {", ".join(["%s"] * len(sql_values))}, NOW())
    ON CONFLICT (user_id) DO UPDATE SET
        {set_query_string};
    """
    
    final_sql_values = [user_id] + sql_values + conflict_values

    # 4. Try to execute the query
    try:
        with db.cursor() as cur:
            cur.execute(sql_query, final_sql_values)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error updating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}"
        )

    return {"status": "success", "message": "Profile updated successfully.", "user_id": user_id}