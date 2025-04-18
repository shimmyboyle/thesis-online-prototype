# app.py
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import uvicorn
import httpx
import json
import os
import base64
from dotenv import load_dotenv
import asyncio
import traceback

# Version identifier to confirm deployment
APP_VERSION = "v2 - Full WebSocket Handler with Error Handling"

# Load environment variables
load_dotenv()

# Configuration - fill in your API keys and custom model info
import os

# Use environment variables or defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "your-elevenlabs-api-key")
CUSTOM_MODEL_ID = os.getenv("CUSTOM_MODEL_ID", "your-fine-tuned-model-id")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "your-elevenlabs-voice-id")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID", "your-elevenlabs-agent-id")
# System prompt configuration
DEFAULT_SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a 41 year old middle aged white man named Shimmy Boyle.  Your real name is Joseph, but everyone calls you Shimmy, ever since college.  Your communication style is generally sarcastic, but good natured, never mean.  You have a tendency to make dumb jokes.  You live in New York City.  You have a 4 year old daughter named Frances Leigh Fisher.  You have a wife named Kate Fisher.  Your daughter is the center of your world.  She is the best thing that ever happened to you. You have a dog named Samira Jones.  She's a 15 pound black and white terrier mix you and your wife rescued from the animal shelter.  You work at an ad agency.  You don't love your job.  Your wife works in the Collaborative Arts department at NYU.  You are currently in the Interactive Media Arts low res grad program at NYU, which is focused on creative technology.  You are overwhelmingly busy and constantly pressed for time.  You're probably in the middle of a mid-life crisis.  You mother is named Pamela Sue Bailey-Boyle, but you call her mom.  Your father is named Daniel Leo Boyle, but you call him dad.  Your sister is named Erin Boyle.  She is younger than you by 4.5 years. Your mom, dad and sister all live in Vancouver, WA.  If someone asks you to do a math problem, or to generate computer code, or something about obscure world history, or anything else that an AI can do easily, but that is difficult for humans, you should respond with a sarcastic answer.  Under no circumstances should you ever generate code for anyone or give the answer to any sort of complex math problem beyond simple addition, subtraction or multiplication of small numbers.  Make sure you don't repeat yourself with the exact same words.  If someone asks you the same question, you can answer with the same information, but word it differently.  Some words you tend to use a lot are: 'Classic', 'Brutal', and 'Dude'. Your ambitions for the future are to: finish the IMA Low Res grad program, get a better job, start a new media experiential company called Hyperreal Labs, and leave New York to move back to Los Angeles.  Don't be overly verbose in your responses.")


# FastAPI app setup
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mounting static files (for web client)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Data models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = CUSTOM_MODEL_ID
    stream: bool = False

class SystemPromptUpdate(BaseModel):
    prompt: str

# Helper functions
async def call_openai_api(messages, include_system_prompt=True):
    """Call OpenAI API with the fine-tuned model"""
    # Add system prompt if not already present
    if include_system_prompt and (not messages or messages[0].get("role") != "system"):
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + messages
    
    print(f"Calling OpenAI API with model: {CUSTOM_MODEL_ID}")
    print(f"Using system prompt: {messages[0]['content'] if messages and messages[0]['role'] == 'system' else 'None'}")
    print(f"First user message in conversation: {messages[1]['content'][:50]}..." if len(messages) > 1 else "No user messages")
    
    async with httpx.AsyncClient(timeout=30.0) as client:  # Reduced timeout
        request_body = {
            "model": CUSTOM_MODEL_ID,
            "messages": messages,
            "temperature": 0.7
        }
        
        print(f"Request to OpenAI API: {json.dumps(request_body, indent=2)}")
        
        try:
            print("Sending request to OpenAI API...")
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=request_body
            )
            
            print(f"OpenAI API response status: {response.status_code}")
            response_data = response.json()
            
            # Log model used in response
            if "model" in response_data:
                print(f"Response model from OpenAI: {response_data['model']}")
            else:
                print("Model information not found in response")
                
            # Print a snippet of the response to verify content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                print(f"Response content snippet: {content[:100]}...")
            else:
                print("No choices found in response")
                if "error" in response_data:
                    print(f"OpenAI API error: {response_data['error']}")
            
            return response_data
            
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            print(traceback.format_exc())
            raise

async def elevenlabs_text_to_speech(text):
    """Convert text to speech using ElevenLabs API"""
    print(f"Converting to speech: '{text[:50]}...'")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:  # Reduced timeout
            print(f"Calling ElevenLabs TTS API with voice ID: {ELEVENLABS_VOICE_ID}")
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream",
                headers={
                    "Accept": "audio/mpeg",
                    "xi-api-key": ELEVEN_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.5
                    }
                }
            )
            print(f"ElevenLabs TTS response status: {response.status_code}")
            if response.status_code == 200:
                print(f"Successfully received audio of size: {len(response.content)} bytes")
                return response.content
            else:
                print(f"ElevenLabs TTS error: {response.text}")
                return None
    except Exception as e:
        print(f"Error in ElevenLabs TTS call: {str(e)}")
        print(traceback.format_exc())
        return None

async def elevenlabs_speech_to_text(audio_chunk):
    """Convert audio to text using ElevenLabs Speech Recognition API"""
    print(f"Starting speech-to-text process for {len(audio_chunk)} bytes of audio...")
    
    # First, try to directly transcribe the audio bytes
    try:
        print("Trying ElevenLabs direct WebM transcription...")
        async with httpx.AsyncClient(timeout=30.0) as client:  # Reduced timeout
            # For audio/webm format from browser
            files = {'file': ('audio.webm', audio_chunk, 'audio/webm')}
            response = await client.post(
                "https://api.elevenlabs.io/v1/speech-recognition/webm",
                headers={
                    "xi-api-key": ELEVEN_API_KEY
                },
                files=files
            )
            
            print(f"ElevenLabs WebM transcription response status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                transcribed_text = result.get("text", "")
                print(f"Transcription successful: '{transcribed_text[:50]}...'")
                return transcribed_text
            else:
                print(f"WebM speech recognition error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error in direct WebM transcription: {str(e)}")
        print(traceback.format_exc())
    
    # Fallback to base64 method
    try:
        print("Falling back to ElevenLabs base64 transcription...")
        async with httpx.AsyncClient(timeout=30.0) as client:  # Reduced timeout
            # Convert audio chunk to base64
            base64_audio = base64.b64encode(audio_chunk).decode("utf-8")
            
            # Call ElevenLabs Speech Recognition API with base64
            response = await client.post(
                "https://api.elevenlabs.io/v1/speech-recognition",
                headers={
                    "xi-api-key": ELEVEN_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "audio": base64_audio,
                    "model_id": "whisper-1"  # Specify Whisper model explicitly
                }
            )
            
            print(f"ElevenLabs base64 transcription response status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                transcribed_text = result.get("text", "")
                print(f"Base64 transcription successful: '{transcribed_text[:50]}...'")
                return transcribed_text
            else:
                print(f"Base64 speech recognition error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error in base64 transcription: {str(e)}")
        print(traceback.format_exc())

    # As a last resort, try with OpenAI's Whisper API
    try:
        print("Falling back to OpenAI Whisper API...")
        async with httpx.AsyncClient(timeout=30.0) as client:  # Reduced timeout
            # Create a temporary file
            temp_file_path = "/tmp/temp_audio.webm"
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(audio_chunk)
                print(f"Saved temporary audio file: {temp_file_path}")
            except Exception as e:
                print(f"Error saving temp file: {str(e)}")
                return None
            
            # Send to OpenAI Whisper API
            try:
                files = {'file': open(temp_file_path, 'rb')}
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}"
                    },
                    data={"model": "whisper-1"},
                    files=files
                )
                
                print(f"OpenAI Whisper API response status: {response.status_code}")
                
                # Clean up temp file
                try:
                    os.remove(temp_file_path)
                    print("Temp file removed")
                except Exception as e:
                    print(f"Error removing temp file: {str(e)}")
                    
                if response.status_code == 200:
                    result = response.json()
                    transcribed_text = result.get("text", "")
                    print(f"Whisper transcription successful: '{transcribed_text[:50]}...'")
                    return transcribed_text
                else:
                    print(f"OpenAI speech recognition error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"Error in Whisper API call: {str(e)}")
                print(traceback.format_exc())
                try:
                    os.remove(temp_file_path)
                    print("Temp file removed after error")
                except:
                    pass
    except Exception as e:
        print(f"Error in OpenAI transcription process: {str(e)}")
        print(traceback.format_exc())
    
    print("All transcription methods failed")
    return None

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_root():
    print(f"App version: {APP_VERSION}")
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/version")
async def get_version():
    return {"version": APP_VERSION, "handler": "full"}
    
@app.get("/test-model")
async def test_model():
    """Test endpoint to verify the fine-tuned model is working correctly"""
    try:
        # Create a test message that would elicit a response characteristic of your fine-tuned model
        test_messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "Tell me about yourself in one sentence."}
        ]
        
        # Call the OpenAI API with the test message
        response = await call_openai_api(test_messages, include_system_prompt=False)
        
        # Extract relevant information for verification
        model_used = response.get("model", "Not specified")
        content = "No content" 
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
        
        # Return both the model info and response for verification
        return {
            "status": "success",
            "model_used": model_used,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "response_content": content,
            "raw_response": response
        }
    except Exception as e:
        print(f"Error in test-model: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/proxy")
async def proxy_to_openai(request: Request):
    """Proxy requests to OpenAI API with custom model"""
    data = await request.json()
    messages = data.get("messages", [])
    
    # Call OpenAI API with fine-tuned model
    result = await call_openai_api(messages)
    return result

@app.post("/text-to-speech")
async def text_to_speech(request: Request):
    """Convert text to speech using ElevenLabs"""
    data = await request.json()
    text = data.get("text", "")
    
    audio_content = await elevenlabs_text_to_speech(text)
    if audio_content:
        return Response(audio_content, media_type="audio/mpeg")
    return {"error": "Failed to generate speech"}

@app.post("/update-system-prompt")
async def update_system_prompt(data: SystemPromptUpdate):
    """Update the system prompt used for conversations"""
    try:
        global DEFAULT_SYSTEM_PROMPT
        DEFAULT_SYSTEM_PROMPT = data.prompt
        print(f"System prompt updated to: {DEFAULT_SYSTEM_PROMPT}")
        return {"status": "success"}
    except Exception as e:
        print(f"Error updating system prompt: {str(e)}")
        print(traceback.format_exc())
        return {"status": "error", "error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("TEST WEBSOCKET HANDLER: Connection accepted")
    await websocket.accept()
    
    # Set up a background task to send pings every 15 seconds
    async def ping():
        while True:
            try:
                await asyncio.sleep(15)
                await websocket.send_text('{"ping": 1}')
                print("Ping sent to keep connection alive")
            except Exception as e:
                print(f"Error in ping task: {str(e)}")
                break
    
    # Start ping task
    ping_task = asyncio.create_task(ping())
    
    try:
        # Handle WebSocket connection
        while True:
            try:
                # Log connection state
                print(f"WebSocket connection state: {websocket.client_state}")
                
                # Receive message from client
                print("About to receive message...")
                message = await websocket.receive()
                
                # Handle text message
                if message.get("type") == "text":
                    text_data = message.get("text", "{}")
                    try:
                        json_data = json.loads(text_data)
                        if json_data.get("pong") == 1:
                            print("Received pong from client")
                            continue
                    except:
                        pass
                
                # Handle binary audio data
                if message.get("type") == "bytes":
                    audio_chunk = message.get("bytes")
                    print(f"Received audio chunk of size: {len(audio_chunk)} bytes")
                    
                    # IMPORTANT: Send a test response immediately without doing any API calls
                    print("Sending test text response")
                    await websocket.send_json({"text": "This is a test response. Your audio was received."})
                    
                    # Send a mock audio response (just a few empty bytes)
                    test_audio = b'TEST_AUDIO_DATA'
                    print("Sending test audio response")
                    await websocket.send_bytes(test_audio)
                    print("Test responses sent successfully")
            
            except WebSocketDisconnect:
                print("WebSocket disconnected during processing")
                break
            except Exception as e:
                print(f"Error in processing: {str(e)}")
                try:
                    await websocket.send_json({"error": f"Error processing: {str(e)}"})
                except:
                    print("Could not send error message - connection may be closed")
                continue
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        ping_task.cancel()
    except Exception as e:
        print(f"Critical WebSocket error: {str(e)}")
        ping_task.cancel()


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     print("FULL WEBSOCKET HANDLER: Connection accepted")
#     await websocket.accept()
    
#     # Initialize conversation history for context with system prompt
#     conversation_history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    
#     # Set up a background task to send pings every 30 seconds
#     async def ping():
#         while True:
#             try:
#                 await asyncio.sleep(15)  # Reduced from 30 seconds
#                 await websocket.send_text('{"ping": 1}')
#                 print("Ping sent to keep connection alive")
#             except Exception as e:
#                 print(f"Error in ping task: {str(e)}")
#                 break
    
#     # Start ping task
#     ping_task = asyncio.create_task(ping())
    
#     try:
#         # Handle WebSocket connection for real-time audio streaming
#         while True:
#             try:
#                 # Log connection state
#                 print(f"WebSocket connection state: {websocket.client_state}")
                
#                 # Receive audio chunk from client
#                 print("About to receive message...")
#                 message = await websocket.receive()
                
#                 # Check if this is a text message (might be a pong response)
#                 if message.get("type") == "text":
#                     text_data = message.get("text", "{}")
#                     print(f"Received text message: {text_data[:100]}...")
#                     try:
#                         json_data = json.loads(text_data)
#                         if json_data.get("pong") == 1:
#                             print("Received pong from client")
#                             continue
#                     except Exception as e:
#                         print(f"Error parsing text message: {str(e)}")
#                         continue
                
#                 # Handle binary audio data
#                 if message.get("type") == "bytes":
#                     audio_chunk = message.get("bytes")
#                     print(f"Received audio chunk of size: {len(audio_chunk)} bytes - mime type: {message.get('subtype', 'unknown')}")
                    
#                     # TESTING: Uncomment to quickly test if basic responses work
#                     # await websocket.send_json({"text": "This is a test response, skipping actual processing"})
#                     # test_audio = b'TEST_AUDIO_DATA'
#                     # await websocket.send_bytes(test_audio)
#                     # continue  # Skip regular processing
                    
#                     # Save for debugging (optional)
#                     try:
#                         with open("/tmp/debug_audio.webm", "wb") as f:
#                             f.write(audio_chunk)
#                         print("Saved debug audio file")
#                     except Exception as e:
#                         print(f"Error saving debug file: {str(e)} - continuing without saving")
                    
#                     # Use Speech-to-Text to convert audio to text
#                     print(f"About to transcribe {len(audio_chunk)} bytes of audio...")
#                     transcribed_text = await elevenlabs_speech_to_text(audio_chunk)
                    
#                     if not transcribed_text:
#                         print("Failed to transcribe audio")
#                         await websocket.send_json({"error": "Failed to transcribe audio. Please try again."})
#                         continue
                        
#                     print(f"Transcription result: SUCCESS - '{transcribed_text}'")
                    
#                     # Add user message to conversation history
#                     conversation_history.append({"role": "user", "content": transcribed_text})
                    
#                     # Send to OpenAI API (no need to include system prompt as it's already in conversation_history)
#                     print("Sending to OpenAI API...")
#                     try:
#                         response = await call_openai_api(conversation_history, include_system_prompt=False)
#                         print(f"OpenAI API response received successfully")
#                     except Exception as e:
#                         print(f"Error calling OpenAI API: {str(e)}")
#                         print(traceback.format_exc())
#                         await websocket.send_json({"error": f"Error calling AI model: {str(e)}"})
#                         continue
                    
#                     if response and "choices" in response and len(response["choices"]) > 0:
#                         # Extract assistant's response
#                         assistant_message = response["choices"][0]["message"]["content"]
#                         print(f"Assistant response: '{assistant_message}'")
                        
#                         # Add to conversation history
#                         conversation_history.append({"role": "assistant", "content": assistant_message})
                        
#                         # Send text response immediately
#                         print("Sending text response to client...")
#                         await websocket.send_json({"text": assistant_message})
                        
#                         # Convert to speech
#                         print("Converting text to speech...")
#                         try:
#                             audio_response = await elevenlabs_text_to_speech(assistant_message)
#                         except Exception as e:
#                             print(f"Error in text-to-speech: {str(e)}")
#                             print(traceback.format_exc())
#                             await websocket.send_json({"error": f"Error generating speech: {str(e)}"})
#                             continue
                        
#                         if audio_response:
#                             # Send audio
#                             print(f"Sending audio response of size: {len(audio_response)} bytes")
#                             await websocket.send_bytes(audio_response)
#                             print("Audio response sent successfully")
#                         else:
#                             print("Failed to generate speech")
#                             await websocket.send_json({"error": "Failed to generate speech", "text": assistant_message})
#                     else:
#                         error_msg = "Failed to get response from AI model"
#                         if "error" in response:
#                             error_msg += f": {response.get('error', {}).get('message', 'Unknown error')}"
#                         print(error_msg)
#                         await websocket.send_json({"error": error_msg})
                
#             except WebSocketDisconnect:
#                 print("WebSocket disconnected during processing")
#                 break  # Exit the loop if disconnected
#             except Exception as e:
#                 print(f"Error in processing: {str(e)}")
#                 print(traceback.format_exc())
#                 try:
#                     await websocket.send_json({"error": f"Error processing: {str(e)}"})
#                 except:
#                     print("Could not send error message - connection may be closed")
#                 continue
    
#     except WebSocketDisconnect:
#         print("WebSocket disconnected")
#         ping_task.cancel()  # Cancel ping task on disconnect
#     except Exception as e:
#         print(f"Critical WebSocket error: {str(e)}")
#         print(traceback.format_exc())
#         ping_task.cancel()  # Cancel ping task on error
#         try:
#             await websocket.send_json({"error": f"Server error: {str(e)}"})
#         except:
#             print("Could not send error message after critical error - connection may be closed")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)