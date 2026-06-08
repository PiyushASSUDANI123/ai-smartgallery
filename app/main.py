import os
import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.db.database import init_db
from app.api.routes import router as api_router
from app.api.routes import expiry_cleanup_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FaceRecognitionAPI")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="SaaS Backend for Multi-Event Facial Recognition Matching with CRM, Proofing, and Superadmin."
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets
# Local static files are no longer served as everything runs on Cloudinary

# Include the API router
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing services and tables...")
    init_db()
    
    # Start auto-expiry scheduler task
    asyncio.create_task(expiry_cleanup_scheduler())

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down backend services...")
