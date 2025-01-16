from fastapi import FastAPI
from routes.register import router
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Face Recognition API", 
              description="API for face recognition and employee registration",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with prefix and tags
app.include_router(router, prefix="/api", tags=["registration"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)