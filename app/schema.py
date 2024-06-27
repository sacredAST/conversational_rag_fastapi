from pydantic import BaseModel
from typing import Optional, List

class Message(BaseModel):
    role: Optional[str] = "system"
    content: Optional[str] = ""

class MessageRequest(BaseModel):
    chatID: Optional[str] = ""
    messages: List[Message]

class ChatResponseModel(BaseModel):
    chatID: Optional[str] = ""
    message: Message