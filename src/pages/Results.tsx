import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, AlertCircle, Info } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Alert, AlertDescription } from '../components/ui/alert';
import { ResultCard } from '../components/ResultCard';
import { RiskGauge } from '../components/RiskGauge';
import { AIPerformanceMetrics } from '../components/AIPerformanceMetrics';
import { ConfidenceVisualization } from '../components/ConfidenceVisualization';
import { MedicalDisclaimer } from '../components/MedicalDisclaimer';
import type { StructuredPredictionResponse, EnhancedHealthResponse } from '../lib/api';

export function Results() {
  const location = useLocation();
  const navigate = useNavigate();
  
  const results = location.state?.results as StructuredPredictionResponse;
  const healthDetails = location.state?.healthDetails as EnhancedHealthResponse;

  if (!results) {
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

  // Extract risk and confidence from structured response
  const finalRisk = results.final.risk;
  const finalConfidence = results.final.confidence;
  const explanation = results.final.explanation;
  const processingTime = results.processing_time_ms;
  
  // Get backend thresholds from health details
  const thresholds = healthDetails?.config ? {
    low: healthDetails.config.risk_thresholds?.low || 0.33,
    high: healthDetails.config.risk_thresholds?.high || 0.66
  } : { low: 0.33, high: 0.66 };

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

        {/* Warnings */}
        {results.warnings && results.warnings.length > 0 && (
          <Alert className="border-orange-200 bg-orange-50">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              <div className="space-y-1">
                {results.warnings.map((warning, index) => (
                  <div key={index}>{warning}</div>
                ))}
              </div>
            </AlertDescription>
          </Alert>
        )}

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
                thresholds={thresholds}
              />
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold text-gray-700 mb-2">Analysis Summary</h3>
                  <p className="text-gray-600 text-sm leading-relaxed">
                    {explanation}
                  </p>
                </div>
                {processingTime && (
                  <div className="text-sm text-gray-500">
                    Processing time: {(processingTime / 1000).toFixed(2)}s
                  </div>
                )}
                
                {/* Backend Configuration Info */}
                {healthDetails && (
                  <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                    <div className="flex items-center gap-2 mb-2">
                      <Info className="h-4 w-4 text-blue-600" />
                      <span className="text-sm font-medium text-blue-800">Backend Configuration</span>
                    </div>
                    <div className="text-xs text-blue-700 space-y-1">
                      <div>Version: {healthDetails.version}</div>
                      <div>Fusion Mode: {results.final.fusion_mode}</div>
                      <div>Risk Thresholds: Low &lt; {(thresholds.low * 100).toFixed(0)}%, High ≥ {(thresholds.high * 100).toFixed(0)}%</div>
                      <div>Models Used: {results.debug?.models_used?.length || 0}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Detailed Results */}
        <div className="grid lg:grid-cols-2 gap-6">
          {results.modalities.map((modality, index) => (
            <ResultCard
              key={`${modality.type}-${index}`}
              title={modality.type === 'face' ? 'Facial Analysis' : 'X-ray Analysis'}
              prediction={modality.label}
              scores={modality.scores}
              originalImage={modality.original_img}
              visualizationImage={modality.visualization}
              foundLabels={modality.found_labels}
              riskLevel={modality.risk as 'low' | 'moderate' | 'high' | 'unknown'}
              confidence={finalConfidence}
              thresholds={thresholds}
              modality={modality}
            />
          ))}
        </div>

        {/* AI Performance Metrics */}
        <AIPerformanceMetrics
          processingTime={processingTime ? processingTime / 1000 : undefined}
          modelCount={results.debug?.models_used?.length || 0}
          ensembleEnabled={healthDetails?.config?.use_ensemble || false}
        />

        {/* Medical Disclaimer */}
        <MedicalDisclaimer />
      </div>
    </div>
  );
}