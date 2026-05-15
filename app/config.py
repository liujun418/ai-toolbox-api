from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/aitoolbox"
    REPLICATE_API_TOKEN: str = ""
    GEMINI_API_KEY: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_S3_ENDPOINT: str = ""  # Empty for AWS S3, Cloudflare R2 URL otherwise
    AWS_S3_BUCKET: str = "ai-toolbox-files"
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    JWT_SECRET_KEY: str = "change-me-in-production"
    FRONTEND_URL: str = "http://localhost:3000"
    RESEND_API_KEY: str = ""

    # Image processing limits
    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB
    MAX_IMAGE_DIMENSION: int = 4096
    MAX_TEXT_LENGTH: int = 8000
    IMAGE_TOOLS_MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    REPLICATE_TIMEOUT_SECONDS: int = 120

    class Config:
        env_file = ".env"


settings = Settings()
