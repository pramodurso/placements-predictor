import os
import psycopg
from fastapi import FastAPI, Depends, HTTPException
from dotenv import load_dotenv
from urllib.parse import quote_plus 
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext

# 1. Load variables from .env file
load_dotenv()

# 2. Create FastAPI app
app = FastAPI()

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: str

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
