"""
File validation utilities for image uploads

Provides comprehensive validation for uploaded images including type checking,
size limits, and security validation with detailed error reporting.
"""

import os
import logging
from typing import List, Optional
from fastapi import UploadFile, HTTPException
from PIL import Image
import io

from config import settings

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

async def validate_image(
    upload: UploadFile,
    allowed_types: Optional[List[str]] = None,
    max_mb: float = 5.0
) -> bytes:
    """
    Validate uploaded image file for security and format compliance
    
    Args:
        upload: FastAPI UploadFile object
        allowed_types: List of allowed MIME types (defaults to settings)
        max_mb: Maximum file size in MB
        
    Returns:
        Image content as bytes
        
    Raises:
        HTTPException: If validation fails
    """
    if not upload or not upload.filename:
        raise HTTPException(status_code=400, detail="No file provided or filename is empty")
    
    allowed_types = allowed_types or settings.ALLOWED_MIME_TYPES
    max_size = int(max_mb * 1024 * 1024)
    
    # Validate file extension
    file_ext = os.path.splitext(upload.filename)[1].lower()
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{file_ext}'. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Validate MIME type
    if upload.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{upload.content_type}'. Allowed: {', '.join(allowed_types)}"
        )
    
    # Read file content for validation
    content = await upload.read()
    file_size = len(content)
    
    # Validate file size
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({max_mb}MB)"
        )
    
    # Validate that it's actually an image
    try:
        with Image.open(io.BytesIO(content)) as img:
            # Verify image can be opened and get basic info
            width, height = img.size
            if width < 32 or height < 32:
                raise HTTPException(status_code=400, detail="Image dimensions too small (minimum 32x32 pixels)")
            if width > 4096 or height > 4096:
                raise HTTPException(status_code=400, detail="Image dimensions too large (maximum 4096x4096 pixels)")
            
            logger.info(f"Validated image: {upload.filename} ({width}x{height}, {file_size} bytes)")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image file: {str(e)}")
    
    return content

def get_safe_filename(filename: str) -> str:
    """
    Generate a safe filename by removing potentially dangerous characters
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem operations
    """
    if not filename:
        return "upload.jpg"
    
    # Remove path components and dangerous characters
    safe_name = os.path.basename(filename)
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._-")
    
    # Ensure filename is not empty and has reasonable length
    if not safe_name or len(safe_name) < 3:
        safe_name = "upload.jpg"
    elif len(safe_name) > 100:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:95] + ext
    
    return safe_name

def validate_proxy_url(url: str) -> bool:
    """
    Validate if URL is allowed for image proxy
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is allowed, False otherwise
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        
        # Check if host is in whitelist
        if parsed.hostname in settings.ALLOWED_PROXY_HOSTS:
            return True
        
        # Check for localhost variations in development
        if settings.DEBUG and parsed.hostname in ['localhost', '127.0.0.1']:
            return True
        
        return False
        
    except Exception:
        return False