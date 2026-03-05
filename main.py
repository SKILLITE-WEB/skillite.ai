from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import requests
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"

# Memory store
sessions = {}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message")
    session_id = data.get("session_id")

    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = {
            "messages": []
        }

    session = sessions[session_id]

    # Save user message
    session["messages"].append({
        "role": "user",
        "content": user_message
    })

    # -------- SMART MASTER PROMPT --------
    system_prompt = """
You are Skillite AI - a smart career mentor for Indian students.

Your Behaviour Rules:

1. First deeply understand the student.
2. Ask only ONE smart question at a time.
3. Detect missing info like:
   - current education or job
   - skill level
   - interests
   - career goal
   - timeline
4. Ask follow-up questions dynamically based on conversation.
5. When you feel you have enough clarity, ask:
   "ok bro, ab mai tera roadmap bana du?"
6. ONLY generate roadmap after user clearly says yes.

After roadmap:
- Continue normal mentor conversation.
- Answer doubts clearly and practically.

Language Rules:
- Use simple daily Hinglish.
- English letters typing only.
- Keep words like skills, roadmap, projects, career, goal in English.
- No heavy Hindi.
- Friendly mentor vibe.
- Adapt to user's language style (Hindi / Marathi / English mix).

Never sound robotic.
Keep responses natural and short unless generating roadmap.
"""

    # Prepare full conversation
    messages_for_api = [
        {"role": "system", "content": system_prompt}
    ] + session["messages"]

    # -------- CALL GROQ API --------
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL_NAME,
            "messages": messages_for_api,
            "temperature": 0.7
        }
    )

    result = response.json()
    ai_reply = result["choices"][0]["message"]["content"]

    # Save AI reply to memory
    session["messages"].append({
        "role": "assistant",
        "content": ai_reply
    })

    return JSONResponse({
        "reply": ai_reply,
        "session_id": session_id
    })
