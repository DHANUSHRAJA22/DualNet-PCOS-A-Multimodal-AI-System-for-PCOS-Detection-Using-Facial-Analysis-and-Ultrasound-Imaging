import React, { useCallback, useState } from 'react';
import { Upload, Camera, Image as ImageIcon, X, FileText, Info } from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { CameraCapture } from './CameraCapture';
import { CameraCapture } from './CameraCapture';
import { fixImageOrientation, type ProcessedImage } from '@/lib/image';
import { toast } from 'sonner';

interface UploadAreaProps {
  id: string;
  label: string;
  subtext: string;
  tips: string;
  onChange: (processedImage: ProcessedImage | null) => void;
  disabled?: boolean;
  acceptedTypes?: string[];
  maxFileSize?: number;
}

export const UploadArea: React.FC<UploadAreaProps> = ({
  id,
  label,
  subtext,
  tips,
  onChange,
  disabled = false,
  acceptedTypes = ['image/jpeg', 'image/png', 'image/webp'],
  maxFileSize = 10 * 1024 * 1024 // 10MB
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [processedImage, setProcessedImage] = useState<ProcessedImage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const validateFile = (file: File): string | null => {
    if (!acceptedTypes.includes(file.type)) {
      return `File type not supported. Please upload: ${acceptedTypes.join(', ')}`;
    }
    if (file.size > maxFileSize) {
      return `File too large. Maximum size: ${Math.round(maxFileSize / (1024 * 1024))}MB`;
    }
    return null;
  };

  const processFile = useCallback(async (file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsProcessing(true);
    setError(null);
    
    try {
      const processed = await fixImageOrientation(file);
      setProcessedImage(processed);
      onChange(processed);
      toast.success(`${label} uploaded successfully`);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to process image';
      setError(message);
      toast.error(`Failed to process ${label.toLowerCase()}: ${message}`);
    } finally {
      setIsProcessing(false);
    }
  }, [onChange, acceptedTypes, maxFileSize, label]);

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
    // Reset input value to allow selecting the same file again
    e.target.value = '';
  }, [processFile]);

  const handleCameraCapture = useCallback((file: File) => {
    processFile(file);
    setCameraOpen(false);
  }, [processFile]);
  const handleCameraCapture = useCallback((file: File) => {
    processFile(file);
    setCameraOpen(false);
  }, [processFile]);

  const clearUpload = useCallback(() => {
    setProcessedImage(null);
    setError(null);
    onChange(null);
  }, [onChange]);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {!processedImage ? (
        <Card
          className={`
            relative border-2 border-dashed transition-all duration-200 cursor-pointer p-8
            ${isDragOver && !disabled 
              ? 'border-blue-400 bg-blue-50 dark:bg-blue-950/20' 
              : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
            }
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
            ${isProcessing ? 'pointer-events-none' : ''}
          `}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="text-center">
            <div className="flex flex-col items-center space-y-4">
              <div className="p-4 bg-gradient-to-br from-purple-100 to-indigo-100 dark:from-purple-900 dark:to-indigo-900 rounded-full">
                {isProcessing ? (
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600" />
                ) : (
                  <Upload className="w-8 h-8 text-purple-600 dark:text-purple-400" />
                )}
              </div>
              
              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-slate-800 dark:text-gray-100">
                  {label}
                </h3>
                <p className="text-sm text-slate-600 dark:text-gray-400">
                  {subtext}
                </p>
                <p className="text-xs text-slate-500 dark:text-gray-500">
                  Supports: JPEG, PNG, WebP (max {Math.round(maxFileSize / (1024 * 1024))}MB)
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-3 w-full max-w-sm">
                <Button
                  variant="outline"
                  className="flex-1"
                  disabled={disabled || isProcessing}
                  onClick={() => document.getElementById(`${id}-file-input`)?.click()}
                >
                  <ImageIcon className="w-4 h-4 mr-2" />
                  {isProcessing ? 'Processing...' : 'Browse Files'}
                </Button>
                
                <Button
                  variant="outline"
                  className="flex-1"
                  disabled={disabled || isProcessing}
                  onClick={() => setCameraOpen(true)}
                >
                  <Camera className="w-4 h-4 mr-2" />
                  Camera
                </Button>
              </div>
              
              <div className="bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 max-w-md">
                <div className="flex items-start gap-2">
                  <Info className="h-4 w-4 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
                  <p className="text-xs text-blue-800 dark:text-blue-200 leading-relaxed">
                    <strong>Tip:</strong> {tips}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <input
            id={`${id}-file-input`}
            type="file"
            accept={acceptedTypes.join(',')}
            onChange={handleFileSelect}
            className="hidden"
            disabled={disabled || isProcessing}
          />
        </Card>
      ) : (
        <Card className="p-6 border-2 border-green-200 bg-gradient-to-br from-green-50 to-emerald-50">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-slate-800 dark:text-gray-100">
                {label} - Ready
              </h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearUpload}
                disabled={disabled || isProcessing}
                className="text-slate-600 hover:text-red-600 hover:bg-red-50"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            <div className="relative">
              <img
                src={processedImage.previewUrl}
                alt="Uploaded preview"
                className="w-full max-h-96 object-contain rounded-lg border-2 border-slate-200 dark:border-gray-700 shadow-lg"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center space-x-3 p-3 bg-white/70 dark:bg-gray-800 rounded-lg border border-slate-200">
                <FileText className="w-5 h-5 text-slate-600 dark:text-gray-400" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-800 dark:text-gray-100 truncate">
                    {processedImage.file.name}
                  </p>
                  <p className="text-xs text-slate-600 dark:text-gray-400">
                    {formatFileSize(processedImage.size)} • {processedImage.file.type}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3 p-3 bg-white/70 dark:bg-gray-800 rounded-lg border border-slate-200">
                <ImageIcon className="w-5 h-5 text-slate-600 dark:text-gray-400" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-800 dark:text-gray-100">
                    {processedImage.width} × {processedImage.height}
                  </p>
                  <p className="text-xs text-slate-600 dark:text-gray-400">
                    {processedImage.orientation}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}
      
      <CameraCapture
        open={cameraOpen}
        onOpenChange={setCameraOpen}
        onCapture={handleCameraCapture}
      />
      
      <CameraCapture
        open={cameraOpen}
        onOpenChange={setCameraOpen}
        onCapture={handleCameraCapture}
      />
    </div>
  );
};