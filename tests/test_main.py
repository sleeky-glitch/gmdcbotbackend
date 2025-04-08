import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "GMDC Chatbot API is running"}

def test_query_valid():
    response = client.post("/query", json={"query": "What is the procedure for applying for a building permit?"})
    assert response.status_code == 200
    assert "response" in response.json()
    assert "success" in response.json()
    assert "references" in response.json()

def test_query_invalid():
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422  # Unprocessable Entity for empty query
