from fastapi import FastAPI
import uvicorn

from api.endpoints import router

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
    # セキュリティ向上: 開発環境でのみ0.0.0.0を使用
    import os

    host = os.environ.get("HOST", "127.0.0.1")
    port_str = os.environ.get("PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    uvicorn.run(app, host=host, port=port)
