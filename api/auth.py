from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

load_dotenv()
# Secret key for signing JWTs (in production: store in .env)
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is required")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Fake user store (replace with DB later)
# Using a properly generated hash for "admin123"
fake_users_db = {
    "admin": {
        "username": "admin",
        # This hash corresponds to "admin123" - generate a new one if needed
        "hashed_password": pwd_context.hash("admin123")  # Generate fresh hash each time
    }
}

# tokenUrl must point to our login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def verify_password(plain_password, hashed_password):
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """Generate password hash."""
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str):
    """Validate user credentials."""
    print(f"Attempting to authenticate user: {username}")  # Debug log
    user = fake_users_db.get(username)
    if not user:
        print(f"User {username} not found")  # Debug log
        return False

    password_valid = verify_password(password, user["hashed_password"])
    print(f"Password valid for {username}: {password_valid}")  # Debug log

    if not password_valid:
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency: Get current logged-in user from JWT."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception
    return user