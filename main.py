import logging

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from config.settings import settings
from routers.health import router as health_router
from routers.home import router as home_router
from routers.uploadfile import router as uploadfile_router
from util.exceptions import IntegrityError
from util.handlers import integrity_error_handler


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title=settings.app_title,
    description=settings.app_description,
    root_path=settings.app_root_path,
)

app.add_exception_handler(IntegrityError, integrity_error_handler)

app.include_router(home_router)
app.include_router(health_router)
app.include_router(uploadfile_router)


def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=f"{settings.app_title} API",
        version=settings.app_version,
        description="A small FastAPI app with routers, custom exception handling, and a customized OpenAPI schema.",
        routes=app.routes,
    )

    openapi_schema["info"]["contact"] = {
        "name": "ICG Sample Team",
        "email": "support@example.com",
    }
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    openapi_schema["servers"] = [
        {"url": settings.app_server_url, "description": settings.app_server_description}
    ]
    openapi_schema["tags"] = [
        {"name": "home", "description": "Endpoints for browser-friendly home pages."},
        {"name": "health", "description": "Endpoints for health and status checks."},
        {"name": "uploadfile", "description": "Endpoints for uploading and deleting files."},
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
