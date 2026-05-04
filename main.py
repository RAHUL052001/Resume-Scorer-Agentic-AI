from fastapi import FastAPI

app=FastAPI()

@app.get("/")
def read_root():
        return {"message": "this is my fastapi tool"}
