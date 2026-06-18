from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import chromadb
from chromadb.config import Settings
import PyPDF2
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="PDF Upload and ChromaDB API")

# ChromaDB configuration
CHROMA_DATA_DIR = "./chroma_data"
Path(CHROMA_DATA_DIR).mkdir(exist_ok=True)

# Initialize ChromaDB client with persistent storage
chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMA_DATA_DIR,
    anonymized_telemetry=False,
)

# Create client
client = chromadb.Client(chroma_settings)

# Create or get collection
collection = client.get_or_create_collection(
    name="pdf_documents",
    metadata={"hnsw:space": "cosine"}
)


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file_content)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "PDF Upload and ChromaDB API is running"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF file and store in ChromaDB"""
    try:
        # Validate file type
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        content = await file.read()
        
        # Extract text from PDF
        text = extract_text_from_pdf(content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF contains no extractable text")
        
        # Split text into chunks (simple approach - by paragraphs)
        chunks = text.split("\n\n")
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        # Store in ChromaDB
        doc_id = file.filename.replace(".pdf", "").replace(" ", "_")
        
        # Add documents to collection
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": file.filename, "chunk_index": i} 
            for i in range(len(chunks))
        ]
        
        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )
        
        logger.info(f"PDF '{file.filename}' uploaded successfully with {len(chunks)} chunks")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF uploaded and indexed successfully",
                "filename": file.filename,
                "total_chunks": len(chunks),
                "doc_id": doc_id
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/search")
async def search_pdf(query: str, top_k: int = 5):
    """Search through uploaded PDF documents using ChromaDB"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Search in ChromaDB
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and len(results["documents"]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "rank": i + 1,
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching PDFs: {str(e)}")


@app.get("/list-documents")
async def list_documents():
    """List all documents in ChromaDB collection"""
    try:
        count = collection.count()
        
        return JSONResponse(
            status_code=200,
            content={
                "total_documents": count,
                "collection_name": "pdf_documents"
            }
        )
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.delete("/clear-collection")
async def clear_collection():
    """Clear all documents from ChromaDB collection"""
    try:
        # Delete the collection and recreate it
        client.delete_collection(name="pdf_documents")
        
        global collection
        collection = client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Collection cleared successfully")
        
        return JSONResponse(
            status_code=200,
            content={"message": "Collection cleared successfully"}
        )
    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing collection: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
