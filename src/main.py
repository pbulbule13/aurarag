"""
Application entrypoint for AuraRAG.

Fill in FastAPI (or your chosen framework) app creation here,
include routers, middlewares, and startup/shutdown events.

Example sketch (to implement later):

    from fastapi import FastAPI

    def create_app() -> FastAPI:
        app = FastAPI()
        # TODO: register routes, middleware, dependencies
        return app

    app = create_app()

If you prefer a different structure (e.g., `src/app/main.py`),
you can move this file accordingly.
"""

# Placeholder for your actual app startup logic
# def create_app():
#     ...

from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
openai_key_exists = bool(os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AuraRAG",
              description="An AI-powered Retrieval-Augmented Generation application.",
              version="0.1.0")

@app.get("/")
async def root():
    return {"message": "Welcome to AuraRAG!",
            "status": "running"
            }


@app.get("/health")
async def health():
    """Detailed Health Check Endpoint"""
    print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
    return {
        "status": "healthy", 
        "openai_configured": openai_key_exists
    }

