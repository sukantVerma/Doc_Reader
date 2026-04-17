from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter(prefix="/home", tags=["home"])


@router.get("/", response_class=HTMLResponse)
async def read_root() -> str:
    return """
    <html>
        <head>
            <title>Sample FastAPI App</title>
        </head>
        <body>
            <h1>FastAPI sample app is running.</h1>
            <p>This is a very basic page served from FastAPI.</p>
        </body>
    </html>
    """
