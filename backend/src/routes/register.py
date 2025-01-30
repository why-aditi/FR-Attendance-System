from fastapi import UploadFile, File, HTTPException, APIRouter, Form, Request
from fastapi.responses import JSONResponse
import aiohttp
import asyncio
import backoff
import logging
from typing import List, Optional
from io import BytesIO

logger = logging.getLogger(__name__)
router = APIRouter()

class APIClient:
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
    async def upload_with_retry(self, employee_id: str, name: str, images: List[UploadFile], 
                              max_retries: int = 3, chunk_size: int = 5):
        """
        Upload images with retry logic and chunking
        """
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Split images into chunks to reduce payload size
            image_chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]
            
            all_responses = []
            for chunk_idx, chunk in enumerate(image_chunks):
                logger.info(f"Processing chunk {chunk_idx + 1}/{len(image_chunks)}")
                
                @backoff.on_exception(
                    backoff.expo,
                    (aiohttp.ClientError, asyncio.TimeoutError),
                    max_tries=max_retries
                )
                async def upload_chunk():
                    data = aiohttp.FormData()
                    data.add_field('employee_id', employee_id)
                    data.add_field('name', name)
                    
                    # Add images to form data
                    for img in chunk:
                        # Reset file pointer
                        await img.seek(0)
                        content = await img.read()
                        data.add_field('images', 
                                     BytesIO(content),
                                     filename=img.filename,
                                     content_type=img.content_type)
                    
                    async with session.post(
                        f"{self.base_url}/register",
                        data=data,
                        timeout=self.timeout
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            text = await response.text()
                            raise HTTPException(
                                status_code=response.status,
                                detail=f"API request failed: {text}"
                            )
                
                try:
                    result = await upload_chunk()
                    all_responses.append(result)
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx + 1} failed after {max_retries} retries: {str(e)}")
                    all_responses.append({
                        "error": str(e),
                        "failed_images": [img.filename for img in chunk]
                    })
            
            return self.combine_responses(all_responses)
    
    def combine_responses(self, responses: List[dict]) -> dict:
        """Combine responses from multiple chunks into a single response"""
        total_successful = 0
        total_failed = 0
        all_failed_images = []
        
        for resp in responses:
            if "error" in resp:
                all_failed_images.extend(resp["failed_images"])
                total_failed += len(resp["failed_images"])
            else:
                total_successful += resp["employee"]["faces_registered"]
                total_failed += resp.get("employee", {}).get("faces_failed", 0)
                if "failed_images" in resp:
                    all_failed_images.extend(resp["failed_images"])
        
        return {
            "message": "Registration completed",
            "summary": {
                "total_processed": total_successful + total_failed,
                "successful": total_successful,
                "failed": total_failed
            },
            "failed_images": all_failed_images if all_failed_images else None
        }

# Example usage in your FastAPI route
@router.post("/register")
async def register_employee(
    request: Request,
    name: str = Form(...),
    employee_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    try:
        api_client = APIClient(base_url="YOUR_API_BASE_URL")
        result = await api_client.upload_with_retry(
            employee_id=employee_id,
            name=name,
            images=images,
            chunk_size=5  # Process 5 images at a time
        )
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )