from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn

from api.endpoints import router
from models.recommendation import StockRecommendation

app = FastAPI(
    title="ClStock - 中期的な推奨銘柄予想API",
    description="初心者向け株式銘柄推奨システム",
    version="1.0.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "ClStock API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
