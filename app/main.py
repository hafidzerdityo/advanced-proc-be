from fastapi import FastAPI, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

# from repositories.postgres.config import engine  # no .database or .metadata anymore
from repositories.postgres.config import db_config  # you'll add this function there
import api.routers.user_management
from logs import logger

load_dotenv(".env")

HOST = os.environ.get("SERVICE_HOST")
PORT = os.environ.get("SERVICE_PORT")
WORKERS = os.environ.get("SERVICE_WORKERS")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create tables at startup
    await db_config.init_db()
    yield
    # nothing to teardown rn

app = FastAPI(
    title="Template Core API",
    description="author: Muhammad Hafidz Erdityo",
    version="0.0.1",
    lifespan=lifespan,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_list = [
        {"loc": err["loc"], "message": err["msg"], "type": err["type"]}
        for err in exc.errors()
    ]
    logger.error(error_list)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"resp_data": None, "resp_msg": error_list}),
    )

# include routers
app.include_router(api.routers.user_management.router, prefix="/api/v1/user_management")
# app.include_router(api.routers.auth.router, prefix="/api/v1/auth")
# app.include_router(api.routers.transaction.router, prefix="/api/v1/trx")

# run with uvicorn main:app






