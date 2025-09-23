import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, AlertCircle } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Alert, AlertDescription } from '../components/ui/alert';
import { ResultCard } from '../components/ResultCard';
import { RiskGauge } from '../components/RiskGauge';
import { AIPerformanceMetrics } from '../components/AIPerformanceMetrics';
import { ConfidenceVisualization } from '../components/ConfidenceVisualization';
import { MedicalDisclaimer } from '../components/MedicalDisclaimer';

interface AnalysisResult {
  face_pred?: string;
  face_scores?: number[];
  face_confidence?: number;
  xray_pred?: string;
  xray_scores?: number[];
  xray_confidence?: number;
  overall_risk?: string;
  combined_explanation?: string;
  final?: {
    risk: string;
    confidence: number;
    explanation: string;
  };
  face_models?: Record<string, number[]>;
  xray_models?: Record<string, number[]>;
  ensemble_score?: number;
  processing_time?: number;
}

export function Results() {
  const location = useLocation();
  const navigate = useNavigate();
  
  const result = location.state?.result as AnalysisResult;
  const uploadedImages = location.state?.uploadedImages as { face?: string; xray?: string };

  if (!result) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-red-600">
              <AlertCircle className="w-5 h-5" />
              No Results Found
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                No analysis results were found. Please go back and upload images for analysis.
              </AlertDescription>
            </Alert>
            <Button 
              onClick={() => navigate('/')} 
              className="w-full"
              variant="outline"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Upload
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Extract risk and confidence from the result
  const finalRisk = result.final?.risk || result.overall_risk || 'unknown';
  const finalConfidence = result.final?.confidence || result.ensemble_score || 0;
  const explanation = result.final?.explanation || result.combined_explanation || '';

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <Button 
            onClick={() => navigate('/')} 
            variant="outline"
            className="flex items-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            New Analysis
          </Button>
          <h1 className="text-2xl font-bold text-gray-800">Analysis Results</h1>
          <div></div>
        </div>

        {/* Risk Overview */}
        <Card>
          <CardHeader>
            <CardTitle>Overall Risk Assessment</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <RiskGauge 
                risk={finalRisk}
                confidence={finalConfidence}
              />
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold text-gray-700 mb-2">Analysis Summary</h3>
                  <p className="text-gray-600 text-sm leading-relaxed">
                    {explanation || 'Analysis completed successfully.'}
                  </p>
                </div>
                {result.processing_time && (
                  <div className="text-sm text-gray-500">
                    Processing time: {result.processing_time.toFixed(2)}s
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Detailed Results */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Face Analysis */}
          {result.face_pred && (
            <ResultCard
              title="Facial Analysis"
              prediction={result.face_pred}
              confidence={result.face_confidence || 0}
              scores={result.face_scores || []}
              image={uploadedImages?.face}
              models={result.face_models}
            />
          )}

          {/* X-ray Analysis */}
          {result.xray_pred && (
            <ResultCard
              title="X-ray Analysis"
              prediction={result.xray_pred}
              confidence={result.xray_confidence || 0}
              scores={result.xray_scores || []}
              image={uploadedImages?.xray}
              models={result.xray_models}
            />
          )}
        </div>

        {/* Confidence Visualization */}
        {(result.face_models || result.xray_models) && (
          <ConfidenceVisualization
            faceModels={result.face_models}
            xrayModels={result.xray_models}
            ensembleScore={result.ensemble_score}
          />
        )}

        {/* AI Performance Metrics */}
        <AIPerformanceMetrics
          processingTime={result.processing_time}
          modelCount={(Object.keys(result.face_models || {}).length + Object.keys(result.xray_models || {}).length)}
          ensembleEnabled={!!(result.face_models || result.xray_models)}
        />

        {/* Medical Disclaimer */}
        <MedicalDisclaimer />
      </div>
    </div>
  );
}