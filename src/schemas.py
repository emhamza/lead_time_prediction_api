from pydantic import BaseModel
from typing import List, Dict, Any, Optional
class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class LoginRequest(BaseModel):
    username: str
    password: str
