from fastapi import Request
from fastapi.responses import JSONResponse

from util.exceptions import IntegrityError


async def integrity_error_handler(request: Request, exc: IntegrityError) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error": "IntegrityError",
            "message": exc.message,
            "path": str(request.url.path),
        },
    )
