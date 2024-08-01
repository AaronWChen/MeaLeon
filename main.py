from fastapi import FastAPI

# use localhost:{port} in browser
# use localhost:{port}/docs to look at the interactive, automatically created documentation

app = FastAPI()


@app.get("/")
async def index() -> dict[str, str]:
    return {"hello": "world"}


@app.get("/about")
async def about() -> str:
    return "About MeaLeon"
