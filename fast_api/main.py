import fastapi, numpy, pandas, pydantic, sklearn, uvicorn

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/whoami")
def whoami():
    return "I am a function"

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, log_level="info")