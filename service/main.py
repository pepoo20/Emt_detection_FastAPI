from fastapi import FastAPI
from service.api.api import main_router

app = FastAPI(project_name = "Emotions Detection")

app.include_router(main_router)