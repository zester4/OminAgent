# app.py (Corrected v9.9 - Handles FormData and Reset)
import uvicorn
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import asyncio
import tempfile
import shutil
import os
import sys # For exit on critical error

# Import the agent class and custom exceptions
# Ensure agent.py is in the same directory or adjust path
try:
    from agent import OmniBotAgent, AgentException, APIKeyError, setup_logging
except ImportError:
    print("ERROR: Could not import OmniBotAgent from agent.py. Make sure agent.py exists and has no syntax errors.", file=sys.stderr)
    sys.exit(1)


# Setup logger for the app
# Using the same setup function from agent.py for consistency
app_logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(title="OmniBot Agent API", version="1.0")

# --- Agent Initialization ---
agent_instance = None
try:
    agent_instance = OmniBotAgent()
    app_logger.info("OmniBotAgent instance created successfully.")
except APIKeyError as e:
    app_logger.critical(f"Agent initialization failed - missing API key: {e}")
    # Allow app to start, but endpoint will return error
except Exception as e:
    app_logger.critical(f"Agent initialization failed: {e}", exc_info=True)
    # Allow app to start, but endpoint will return error

# --- SSE Streaming Endpoint (Corrected for FormData/File Upload) ---
@app.post("/chat_stream")
async def chat_stream_endpoint(
    message: str = Form(...), # Get message from form data
    file: UploadFile | None = File(None) # Get optional file upload
):
    # Check if agent initialized properly during startup
    if not agent_instance:
        async def error_stream(): yield f"event: error\ndata: {json.dumps({'message': 'Agent not initialized. Check server logs.'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    temp_file_path = None
    file_identifier_for_agent = None

    try:
        # --- Handle File Upload ---
        if file:
            if not file.filename:
                 async def error_stream(): yield f"event: error\ndata: {json.dumps({'message': 'Uploaded file is missing a filename.'})}\n\n"
                 return StreamingResponse(error_stream(), media_type="text/event-stream")

            app_logger.info(f"Received file upload request: {file.filename} (type: {file.content_type}, size: {file.size})")
            # Basic size check (e.g., 50MB limit) - adjust as needed
            MAX_FILE_SIZE = 50 * 1024 * 1024
            if file.size > MAX_FILE_SIZE:
                 async def error_stream(): yield f"event: error\ndata: {json.dumps({'message': f'File size exceeds limit ({MAX_FILE_SIZE // 1024 // 1024}MB).'})}\n\n"
                 return StreamingResponse(error_stream(), media_type="text/event-stream")

            suffix = Path(file.filename).suffix.lower()
            # Consider adding more robust file type validation if needed
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="upload_") as temp_file:
                # Read the file content chunk by chunk
                size = 0
                while content := await file.read(1024 * 1024): # Read in 1MB chunks
                     temp_file.write(content)
                     size += len(content)
                     if size > MAX_FILE_SIZE: # Check size again during read
                          os.unlink(temp_file.name) # Clean up partial file
                          async def error_stream(): yield f"event: error\ndata: {json.dumps({'message': f'File size exceeds limit ({MAX_FILE_SIZE // 1024 // 1024}MB).'})}\n\n"
                          return StreamingResponse(error_stream(), media_type="text/event-stream")

                temp_file_path = temp_file.name # Get the path AFTER writing is done
                file_identifier_for_agent = temp_file_path # Agent uses the path
            app_logger.info(f"File '{file.filename}' saved temporarily to '{temp_file_path}'")
        # -------------------------

        async def event_stream():
            # Use the agent's async stream processing method
            async for update in agent_instance.process_message_stream(message, file_identifier_for_agent):
                event_type = update.get("type", "message")
                # Ensure data payload is always a valid JSON string
                try:
                    data_payload = json.dumps(update.get("data"))
                except TypeError:
                    # Handle cases where data might not be directly JSON serializable
                    data_payload = json.dumps(str(update.get("data")))
                yield f"event: {event_type}\ndata: {data_payload}\n\n"
                await asyncio.sleep(0.01) # Yield control briefly

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
         app_logger.error(f"Error processing chat request: {e}", exc_info=True)
         # Ensure error message sent to client is JSON serializable
         error_msg = str(e)
         async def error_stream(): yield f"event: error\ndata: {json.dumps({'message': f'Server error: {error_msg}'})}\n\n"
         return StreamingResponse(error_stream(), media_type="text/event-stream")

    finally:
        # --- Clean up temporary file ---
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                app_logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_err:
                app_logger.error(f"Error cleaning up temporary file {temp_file_path}: {cleanup_err}")


# --- Reset Chat Endpoint ---
@app.post("/reset_chat")
async def reset_chat_endpoint():
    if not agent_instance:
        raise HTTPException(status_code=500, detail="Agent not initialized.")
    try:
        if hasattr(agent_instance, 'reset_memory'):
            agent_instance.reset_memory() # Call the reset method in agent.py
            app_logger.info("Agent memory reset successfully via API.")
            return JSONResponse({"status": "success", "message": "Chat session reset."})
        else:
             app_logger.warning("Agent class does not have a 'reset_memory' method.")
             return JSONResponse({"status": "warning", "message": "Agent reset method not implemented."}, status_code=501)
    except Exception as e:
        app_logger.error(f"Error resetting agent memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset chat: {str(e)}")


# --- Serve Static Files (HTML/CSS/JS) ---
static_dir = Path("static")
if not static_dir.is_dir():
    app_logger.warning("Static directory 'static/' not found. Attempting to create.")
    try: static_dir.mkdir(exist_ok=True) # Use exist_ok=True
    except OSError as e: app_logger.error(f"Could not create static directory: {e}")

index_html_path = static_dir / "index.html"
if not index_html_path.is_file():
     app_logger.warning("static/index.html not found. Web interface will not load.")

# Use check=False to prevent FastAPI error if dir doesn't exist on startup
app.mount("/static", StaticFiles(directory=static_dir, check_dir=False), name="static")


# --- Root Endpoint to Serve HTML ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if not index_html_path.is_file():
         app_logger.error("static/index.html not found! Cannot serve frontend.")
         return HTMLResponse(content="<html><body><h1>OmniAgent Error</h1><p>Error: Frontend file (static/index.html) not found. Please ensure it exists.</p></body></html>", status_code=404)
    return HTMLResponse(content=index_html_path.read_text(), status_code=200)


# --- Run the app (for local development/Codespaces) ---
if __name__ == "__main__":
    print("Starting OmniBot FastAPI server (v9.8)...")
    print("Access the chat interface at http://127.0.0.1:8000 (or forwarded port)")
    # Use reload=True for development convenience
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)