import os
from fastapi import FastAPI
from app.routers.auth import router as auth_router
from app.routers.image import router as image_router
from app.routers.analytics import router as analytics_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        os.getenv("FRONTEND_ORIGIN", "http://localhost"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(image_router)
app.include_router(analytics_router)

@app.get("/")
def home():
    return {
        "message" : "VisionInspect AI Backend Running"
    }
