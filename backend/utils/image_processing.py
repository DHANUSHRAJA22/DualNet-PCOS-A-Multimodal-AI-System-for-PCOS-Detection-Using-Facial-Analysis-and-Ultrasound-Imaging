"""
Image processing utilities for medical image analysis

Handles image preprocessing, format conversion, orientation correction,
and preparation for deep learning model inference.
"""

import os
import uuid
import logging
from typing import Dict, Any, Tuple
from PIL import Image, ImageOps
from fastapi import UploadFile
import aiofiles
import numpy as np
from config import settings
from utils.validators import validate_file_size, get_safe_filename, ValidationError

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Comprehensive image processing for medical image analysis
    
    Handles image upload, validation, preprocessing, format conversion,
    and preparation for various deep learning model architectures.
    """
    
    def __init__(self):
        """Initialize image processor with configuration"""
        self.upload_dir = settings.UPLOAD_DIR
        self.results_dir = settings.RESULTS_DIR
        self.image_quality = settings.IMAGE_QUALITY
        
        logger.info(f"Initialized ImageProcessor: upload_dir={self.upload_dir}")
    
    async def process_face_image(self, upload_file: UploadFile) -> Dict[str, Any]:
        """
        Process uploaded facial image for model inference
        
        Handles file saving, orientation correction, resizing for different model
        architectures, and format standardization.
        
        Args:
            upload_file: FastAPI UploadFile object containing facial image
            
        Returns:
            Dictionary containing:
                - original_path: Path to original uploaded file
                - processed_path: Path to processed image
                - saved_path: Public URL path for frontend
                - filename: Generated filename
                - width, height: Image dimensions
                - mode: Image color mode
                - array: Preprocessed numpy array for model input (TODO)
                
        Raises:
            ValidationError: If image processing fails
            RuntimeError: If file operations fail
        """
        
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            safe_filename = get_safe_filename(upload_file.filename)
            name, ext = os.path.splitext(safe_filename)
            filename = f"face_{file_id}_{name}.jpg"
            file_path = os.path.join(self.upload_dir, filename)
            
            logger.info(f"Processing face image: {upload_file.filename} -> {filename}")
            
            # Save uploaded file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await upload_file.read()
                await f.write(content)
            
            # Validate file size after saving
            validate_file_size(file_path)
            
            # Process image with PIL
            with Image.open(file_path) as img:
                original_width, original_height = img.size
                
                # Fix orientation using EXIF data
                img = ImageOps.exif_transpose(img)
                
                # Convert to RGB if needed (removes alpha channel, handles grayscale)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize for face model input (typically 224x224)
                img_resized = img.resize(settings.FACE_IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                # Save processed image
                processed_path = os.path.join(self.upload_dir, f"processed_{filename}")
                img_resized.save(processed_path, 'JPEG', quality=self.image_quality)
                
                # TODO: Convert to numpy array for model input
                # Example:
                # img_array = np.array(img_resized, dtype=np.float32) / 255.0  # Normalize to [0,1]
                # img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                # 
                # For different models, you might need different preprocessing:
                # - EfficientNet: ImageNet normalization
                # - ResNet: ImageNet normalization  
                # - VGG: ImageNet normalization
                
                result = {
                    "original_path": file_path,
                    "processed_path": processed_path,
                    "saved_path": f"/static/uploads/{filename}",
                    "filename": filename,
                    "width": original_width,
                    "height": original_height,
                    "processed_width": settings.FACE_IMAGE_SIZE[0],
                    "processed_height": settings.FACE_IMAGE_SIZE[1],
                    "mode": img.mode,
                    "file_size_bytes": os.path.getsize(file_path)
                    # "array": img_array  # TODO: Add when implementing real models
                }
                
                logger.info(f"Face image processed successfully: {filename}")
                return result
                
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Face image processing failed: {str(e)}")
            # Clean up any created files
            self._cleanup_files([file_path, processed_path])
            raise RuntimeError(f"Face image processing failed: {str(e)}")
    
    async def process_xray_image(self, upload_file: UploadFile) -> Dict[str, Any]:
        """
        Process uploaded X-ray image for model inference
        
        Prepares images for both YOLO object detection and Vision Transformer
        classification with appropriate resizing and preprocessing.
        
        Args:
            upload_file: FastAPI UploadFile object containing X-ray image
            
        Returns:
            Dictionary containing:
                - original_path: Path to original uploaded file
                - yolo_path: Path to YOLO-preprocessed image (640x640)
                - vit_path: Path to ViT-preprocessed image (224x224)
                - saved_path: Public URL path for frontend
                - filename: Generated filename
                - width, height: Original image dimensions
                - mode: Image color mode
                - yolo_array, vit_array: Preprocessed arrays for models (TODO)
                
        Raises:
            ValidationError: If image processing fails
            RuntimeError: If file operations fail
        """
        
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            safe_filename = get_safe_filename(upload_file.filename)
            name, ext = os.path.splitext(safe_filename)
            filename = f"xray_{file_id}_{name}.jpg"
            file_path = os.path.join(self.upload_dir, filename)
            
            logger.info(f"Processing X-ray image: {upload_file.filename} -> {filename}")
            
            # Save uploaded file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await upload_file.read()
                await f.write(content)
            
            # Validate file size after saving
            validate_file_size(file_path)
            
            # Process image with PIL
            with Image.open(file_path) as img:
                original_width, original_height = img.size
                
                # Fix orientation using EXIF data
                img = ImageOps.exif_transpose(img)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Prepare for YOLO (typically 640x640)
                img_yolo = img.resize(settings.XRAY_YOLO_SIZE, Image.Resampling.LANCZOS)
                yolo_path = os.path.join(self.upload_dir, f"yolo_{filename}")
                img_yolo.save(yolo_path, 'JPEG', quality=self.image_quality)
                
                # Prepare for Vision Transformer (typically 224x224 or 384x384)
                img_vit = img.resize(settings.XRAY_VIT_SIZE, Image.Resampling.LANCZOS)
                vit_path = os.path.join(self.upload_dir, f"vit_{filename}")
                img_vit.save(vit_path, 'JPEG', quality=self.image_quality)
                
                # TODO: Convert to numpy arrays for model input
                # Example:
                # yolo_array = np.array(img_yolo, dtype=np.float32) / 255.0
                # vit_array = np.array(img_vit, dtype=np.float32) / 255.0
                # vit_array = np.expand_dims(vit_array, axis=0)  # Add batch dimension
                #
                # For YOLO, you might need different preprocessing:
                # - Normalize to [0,1] or keep [0,255] depending on model training
                # - Handle different input formats (RGB vs BGR)
                #
                # For ViT, standard preprocessing:
                # - Normalize to [0,1] or ImageNet normalization
                # - Proper tensor format for transformer input
                
                result = {
                    "original_path": file_path,
                    "yolo_path": yolo_path,
                    "vit_path": vit_path,
                    "saved_path": f"/static/uploads/{filename}",
                    "filename": filename,
                    "width": original_width,
                    "height": original_height,
                    "yolo_width": settings.XRAY_YOLO_SIZE[0],
                    "yolo_height": settings.XRAY_YOLO_SIZE[1],
                    "vit_width": settings.XRAY_VIT_SIZE[0],
                    "vit_height": settings.XRAY_VIT_SIZE[1],
                    "mode": img.mode,
                    "file_size_bytes": os.path.getsize(file_path)
                    # "yolo_array": yolo_array,  # TODO: Add when implementing real models
                    # "vit_array": vit_array     # TODO: Add when implementing real models
                }
                
                logger.info(f"X-ray image processed successfully: {filename}")
                return result
                
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"X-ray image processing failed: {str(e)}")
            # Clean up any created files
            self._cleanup_files([file_path, yolo_path, vit_path])
            raise RuntimeError(f"X-ray image processing failed: {str(e)}")
    
    def _cleanup_files(self, file_paths: List[str]) -> None:
        """
        Clean up temporary files after processing failure
        
        Args:
            file_paths: List of file paths to remove
        """
        for path in file_paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Cleaned up file: {path}")
            except Exception as e:
                logger.warning(f"Could not remove file {path}: {str(e)}")
    
    async def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old uploaded and processed files
        
        Args:
            max_age_hours: Maximum age of files to keep (in hours)
            
        Returns:
            Number of files cleaned up
        """
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        for directory in [self.upload_dir, self.results_dir]:
            if not os.path.exists(directory):
                continue
                
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        cleaned_count += 1
                        logger.debug(f"Cleaned up old file: {filename}")
                        
                except Exception as e:
                    logger.warning(f"Could not clean up file {filename}: {str(e)}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old files")
        
        return cleaned_count