from fastapi import APIRouter

test_router = APIRouter()

@test_router.get("/test/")  # Note the forward slash before "test/"
def testing():
    return {"testing": "testing"}
