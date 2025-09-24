import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, AlertCircle, Info, User } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { ResultCard } from '@/components/ResultCard';
import { RiskGauge } from '@/components/RiskGauge';
import { AIPerformanceMetrics } from '@/components/AIPerformanceMetrics';
import { MedicalDisclaimer } from '@/components/MedicalDisclaimer';
import type { StructuredPredictionResponse, EnhancedHealthResponse } from '@/lib/api';
import { isStructuredResponse } from '@/lib/api';

export function Results() {
  const location = useLocation();
  const navigate = useNavigate();
  
  const rawResults = location.state?.results;
  const healthDetails = location.state?.healthDetails as EnhancedHealthResponse;
  
  // Extract final confidence from results
  const finalConfidence = rawResults?.final?.confidence || 0;

  if (!rawResults) {
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

  // Ensure we have structured results
  let results: StructuredPredictionResponse;
  
  try {
    if (isStructuredResponse(rawResults)) {
      results = rawResults;
    } else {
      // Convert legacy format to structured format
      results = {
        ok: rawResults?.ok || false,
        modalities: [],
        final: {
          risk: rawResults?.overall_risk || 'unknown',
          confidence: 0.5,
          explanation: rawResults?.combined || 'Analysis completed',
          fusion_mode: 'legacy'
        },
        warnings: [],
        processing_time_ms: 0,
        debug: {}
      };
      
      // Add face modality if present
      if (rawResults?.face_pred) {
        results.modalities.push({
          type: 'face',
          label: rawResults.face_pred,
          scores: rawResults.face_scores || [],
          risk: rawResults.face_risk || 'unknown',
          original_img: rawResults.face_img
        });
      }
      
      // Add xray modality if present
      if (rawResults?.xray_pred) {
        results.modalities.push({
          type: 'xray',
          label: rawResults.xray_pred,
          scores: [],
          risk: rawResults.xray_risk || 'unknown',
          original_img: rawResults.xray_img,
          visualization: rawResults.yolo_vis,
          found_labels: rawResults.found_labels
        });
      }
    }
  } catch (error) {
    console.error('Error processing results:', error);
    // Fallback to minimal results structure
    results = {
      ok: false,
      modalities: [],
      final: {
        risk: 'unknown',
        confidence: 0,
        explanation: 'Error processing results',
        fusion_mode: 'error'
      },
      warnings: ['Error processing analysis results'],
      processing_time_ms: 0,
      debug: {}
    };
  }

  // Extract risk and confidence from structured response
  const finalRisk = results.final.risk;
  const finalConfidence = results.final.confidence;
  const explanation = results.final.explanation;
  const processingTime = results.processing_time_ms;
  
  // Get backend thresholds from health details
  const thresholds = healthDetails?.config ? {
    low: healthDetails.config.risk_thresholds?.low ?? 0.33,
    high: healthDetails.config.risk_thresholds?.high ?? 0.66
  } : { low: 0.33, high: 0.66 };

  // Find face modality for gender display
  const faceModality = results.modalities.find(m => m.type === 'face');
  const xrayModality = results.modalities.find(m => m.type === 'xray');

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

        {/* Gender Detection Display */}
        {faceModality?.gender && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="w-5 h-5" />
                Gender Detection
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-2">
                    Detected Gender: <span className="font-semibold capitalize">{faceModality.gender.label}</span>
                  </p>
                  <div className="flex gap-4 text-sm">
                    <div>Male: {(faceModality.gender.male * 100).toFixed(1)}%</div>
                    <div>Female: {(faceModality.gender.female * 100).toFixed(1)}%</div>
                  </div>
                </div>
                <Badge 
                  variant="outline" 
                  className={`capitalize ${
                    faceModality.gender.label === 'female' 
                      ? 'border-pink-200 text-pink-700 bg-pink-50' 
                      : 'border-blue-200 text-blue-700 bg-blue-50'
                  }`}
                >
                  {faceModality.gender.label} ({(Math.max(faceModality.gender.male, faceModality.gender.female) * 100).toFixed(1)}%)
                </Badge>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Risk Overview */}
        <Card>
          <CardHeader>
            <CardTitle>Overall Risk Assessment</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <RiskGauge 
                riskLevel={finalRisk as 'low' | 'moderate' | 'high' | 'unknown'}
                confidence={finalConfidence > 1 ? finalConfidence : finalConfidence * 100}
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
                      {results.debug?.weights && Object.keys(results.debug.weights).length > 0 && (
                        <div>
                          Ensemble Weights: 
                          {Object.entries(results.debug.weights).map(([modality, weights]: [string, any]) => (
                            <div key={modality} className="ml-2">
                              {modality}: {Object.entries(weights).map(([model, weight]: [string, any]) => 
                                `${model}=${weight.toFixed(3)}`
                              ).join(', ')}
                            </div>
                          ))}
                        </div>
                      )}
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
            <div key={`${modality.type}-${index}`}>
              {/* Handle missing X-ray models gracefully */}
              {modality.type === 'xray' && (!modality.per_model || Object.keys(modality.per_model).length === 0) ? (
                <Card className="border-2 border-amber-200 bg-gradient-to-br from-amber-50 to-orange-50">
                  <CardHeader>
                    <CardTitle className="text-xl font-bold text-slate-800">X-ray Analysis</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Alert className="border-amber-200 bg-amber-50">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        X-ray PCOS models unavailable—imaging analysis skipped.
                      </AlertDescription>
                    </Alert>
                    
                    {/* Show YOLO detections if available */}
                    {modality.detections && modality.detections.length > 0 && (
                      <div>
                        <h4 className="font-semibold mb-3 text-slate-800">Object Detections</h4>
                        <div className="space-y-2">
                          {modality.detections.map((detection, detIndex) => (
                            <div key={detIndex} className="bg-white/70 p-3 rounded-lg border border-slate-200">
                              <div className="flex justify-between items-center">
                                <span className="capitalize font-medium">{detection.label}</span>
                                <Badge variant="outline">
                                  {(detection.conf * 100).toFixed(1)}% confidence
                                </Badge>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Show visualization if available */}
                    {modality.visualization && (
                      <div>
                        <h4 className="font-semibold mb-3 text-slate-800">YOLO Detection Visualization</h4>
                        <img
                          src={modality.visualization}
                          alt="YOLO Detection"
                          className="w-full max-h-64 object-contain rounded-lg border-2 border-slate-200 bg-white shadow-lg"
                          loading="lazy"
                        />
                      </div>
                    )}
                    
                    {modality.original_img && (
                      <div>
                        <h4 className="font-semibold mb-3 text-slate-800">Uploaded Image</h4>
                        <img
                          src={modality.original_img}
                          alt="X-ray"
                          className="w-full max-h-64 object-contain rounded-lg border-2 border-slate-200 bg-white shadow-lg"
                          loading="lazy"
                        />
                      </div>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <ResultCard
                  title={modality.type === 'face' ? 'Facial Analysis' : 'X-ray Analysis'}
                  prediction={modality.label}
                  scores={modality.scores}
                  originalImage={modality.original_img}
                  visualizationImage={modality.visualization}
                  foundLabels={modality.found_labels}
                  riskLevel={modality.risk as 'low' | 'moderate' | 'high' | 'unknown'}
                  confidence={finalConfidence > 1 ? finalConfidence : finalConfidence * 100}
                  thresholds={thresholds}
                  modality={modality}
                />
              )}
            </div>
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