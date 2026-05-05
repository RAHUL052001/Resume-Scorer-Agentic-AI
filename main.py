from fastapi import FastAPI

app=FastAPI()

@app.get("/")
def read_root():
        return {"message": "this is my fastapi tool"}


@app.get("/second API")
def read_func():
        return {
                "message":"Second api for the response"
        }
