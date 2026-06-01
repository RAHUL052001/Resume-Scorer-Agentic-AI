from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="Resume Selector API")

class TextRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/greet")
def greet(name: str = Query("world")):
    return {"message": f"hello, {name}"}

@app.post("/reverse")
def reverse_text(req: TextRequest):
    return {"original": req.text, "reversed": req.text[::-1]}