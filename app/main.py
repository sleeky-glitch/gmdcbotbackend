from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import QueryRequest, QueryResponse
from app.services.pinecone_service import PineconeService
from app.services.openai_service import OpenAIService

app = FastAPI(title="GMDC Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

pinecone_service = PineconeService()
openai_service = OpenAIService()

@app.get("/")
async def root():
    return {"message": "GMDC Chatbot API is running"}

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    print("Received request:", request)  # Log the received request
    try:
        # Query Pinecone for relevant context
        pinecone_result = await pinecone_service.query_documents(request.query)
        context, references = pinecone_service.extract_context(pinecone_result)

        # Generate response using OpenAI
        response_text = await openai_service.generate_response(context, request.query)

        return QueryResponse(
            response=response_text,
            success=True,
            references=references
        )

    except Exception as e:
        print("Error occurred:", str(e))  # Log the error
        raise HTTPException(status_code=500, detail=str(e))
