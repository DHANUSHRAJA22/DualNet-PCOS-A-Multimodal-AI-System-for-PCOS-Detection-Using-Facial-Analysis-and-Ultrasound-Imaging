import React, { useCallback, useState } from 'react';
import { Upload, Camera, Image as ImageIcon, X, FileText } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface UploadAreaProps {
  onImageUpload: (file: File, preview: string) => void;
  onCameraCapture?: () => void;
  disabled?: boolean;
  acceptedTypes?: string[];
  maxFileSize?: number;
}

export const UploadArea: React.FC<UploadAreaProps> = ({
  onImageUpload,
  onCameraCapture,
  disabled = false,
  acceptedTypes = ['image/jpeg', 'image/png', 'image/webp'],
  maxFileSize = 10 * 1024 * 1024 // 10MB
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const validateFile = (file: File): string | null => {
    if (!acceptedTypes.includes(file.type)) {
      return `File type not supported. Please upload: ${acceptedTypes.join(', ')}`;
    }
    if (file.size > maxFileSize) {
      return `File too large. Maximum size: ${Math.round(maxFileSize / (1024 * 1024))}MB`;
    }
    return null;
  };

  const processFile = useCallback((file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setPreview(result);
      setUploadedFile(file);
      onImageUpload(file, result);
    };
    reader.readAsDataURL(file);
  }, [onImageUpload, acceptedTypes, maxFileSize]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragOver(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      processFile(files[0]);
    }
  }, [disabled, processFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  }, [processFile]);

  const clearUpload = useCallback(() => {
    setPreview(null);
    setUploadedFile(null);
    setError(null);
  }, []);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {!preview ? (
        <Card
          className={`
            relative border-2 border-dashed transition-all duration-200 cursor-pointer
            ${isDragOver && !disabled 
              ? 'border-blue-400 bg-blue-50 dark:bg-blue-950/20' 
              : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
            }
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="p-8 text-center">
            <div className="flex flex-col items-center space-y-4">
              <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-full">
                <Upload className="w-8 h-8 text-gray-600 dark:text-gray-400" />
              </div>
              
              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Upload Medical Image
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Drag and drop your image here, or click to browse
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-500">
                  Supports: JPEG, PNG, WebP (max {Math.round(maxFileSize / (1024 * 1024))}MB)
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-3 w-full max-w-sm">
                <Button
                  variant="outline"
                  className="flex-1"
                  disabled={disabled}
                  onClick={() => document.getElementById('file-input')?.click()}
                >
                  <ImageIcon className="w-4 h-4 mr-2" />
                  Browse Files
                </Button>
                
                {onCameraCapture && (
                  <Button
                    variant="outline"
                    className="flex-1"
                    disabled={disabled}
                    onClick={onCameraCapture}
                  >
                    <Camera className="w-4 h-4 mr-2" />
                    Camera
                  </Button>
                )}
              </div>
            </div>
          </div>

          <input
            id="file-input"
            type="file"
            accept={acceptedTypes.join(',')}
            onChange={handleFileSelect}
            className="hidden"
            disabled={disabled}
          />
        </Card>
      ) : (
        <Card className="p-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Uploaded Image
              </h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearUpload}
                disabled={disabled}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            <div className="relative">
              <img
                src={preview}
                alt="Uploaded preview"
                className="w-full max-h-96 object-contain rounded-lg border border-gray-200 dark:border-gray-700"
              />
            </div>

            {uploadedFile && (
              <div className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <FileText className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                    {uploadedFile.name}
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    {formatFileSize(uploadedFile.size)} • {uploadedFile.type}
                  </p>
                </div>
              </div>
            )}
          </div>
        </Card>
      )}

      {error && (
        <div className="mt-4 p-3 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}
    </div>
  );
};