"use client"

import { useState, useEffect } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Separator } from "@/components/ui/separator"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from "recharts"
import { Activity, Heart, Droplets, AlertTriangle, CheckCircle, XCircle, Info, Stethoscope, Brain, TrendingUp, Users, Clock, Shield } from "lucide-react"
import axios from "axios"

// API Configuration
const API_BASE_URL = "http://localhost:8000"

// Enhanced Patient Data Interface
interface EnhancedPatientData {
  // Demographics (Required)
  Age: number
  Gender: number
  
  // Core Vital Signs
  HR?: number
  O2Sat?: number
  Temp?: number
  SBP?: number
  MAP?: number
  DBP?: number
  Resp?: number
  
  // Laboratory Values
  Glucose?: number
  BUN?: number
  Creatinine?: number
  WBC?: number
  Hct?: number
  Hgb?: number
  Platelets?: number
  Lactate?: number
  
  // Additional Lab Values
  Bilirubin_total?: number
  AST?: number
  Alkalinephos?: number
  Calcium?: number
  Chloride?: number
  Magnesium?: number
  Phosphate?: number
  Potassium?: number
  
  // Blood Gas and Respiratory
  pH?: number
  PaCO2?: number
  BaseExcess?: number
  HCO3?: number
  FiO2?: number
  
  // Temporal
  ICULOS?: number
  
  // Clinical Context
  patient_id?: string
  unit?: string
}

// Enhanced Prediction Response
interface EnhancedPredictionResponse {
  patient_id: string
  timestamp: string
  survival_probability: number
  mortality_probability: number
  risk_level: "low" | "moderate" | "high" | "critical"
  risk_score: number
  prediction: string
  confidence: number
  shap_explanations?: Record<string, {
    value: number
    shap_contribution: number
    impact: string
  }>
  clinical_alerts: string[]
  recommendations: string[]
  model_version: string
  processing_time_ms: number
}

// Feature Importance Interface
interface FeatureImportance {
  feature: string
  importance: number
  description: string
}

// Model Info Interface
interface ModelInfo {
  model_type: string
  model_version: string
  feature_count: number
  total_features: number
  has_shap: boolean
  performance_metrics: Record<string, any>
  preprocessing_info: {
    enhanced_preprocessing: boolean
    selected_features_count: number
    feature_selection_method: string
    clinical_priority_features: boolean
    temporal_features: boolean
    physiological_features: boolean
  }
  supported_features: {
    clinical_scores: string[]
    physiological_indices: string[]
    clinical_alerts: boolean
    batch_prediction: boolean
    shap_explanations: boolean
  }
  api_version: string
}

// Sample Enhanced Patient Data
const ENHANCED_SAMPLE_PATIENT: EnhancedPatientData = {
  Age: 68,
  Gender: 1,
  HR: 105,
  O2Sat: 92,
  Temp: 38.7,
  SBP: 88,
  MAP: 62,
  DBP: 55,
  Resp: 26,
  Glucose: 145,
  BUN: 28,
  Creatinine: 1.8,
  WBC: 16.2,
  Hct: 32,
  Hgb: 10.5,
  Platelets: 95,
  Lactate: 3.8,
  pH: 7.32,
  HCO3: 18,
  ICULOS: 18,
  patient_id: "demo_patient_001",
  unit: "ICU"
}

// Risk level configurations with enhanced styling
const ENHANCED_RISK_LEVELS = {
  low: {
    color: "bg-green-500 text-white",
    textColor: "text-green-600",
    bgColor: "bg-green-50 border border-green-200",
    iconColor: "text-green-500",
    gradient: "from-green-400 to-green-600"
  },
  moderate: {
    color: "bg-yellow-500 text-white",
    textColor: "text-yellow-600", 
    bgColor: "bg-yellow-50 border border-yellow-200",
    iconColor: "text-yellow-500",
    gradient: "from-yellow-400 to-yellow-600"
  },
  high: {
    color: "bg-orange-500 text-white",
    textColor: "text-orange-600",
    bgColor: "bg-orange-50 border border-orange-200", 
    iconColor: "text-orange-500",
    gradient: "from-orange-400 to-orange-600"
  },
  critical: {
    color: "bg-red-500 text-white",
    textColor: "text-red-600",
    bgColor: "bg-red-50 border border-red-200",
    iconColor: "text-red-500",
    gradient: "from-red-400 to-red-600"
  }
}

export default function EnhancedSepsisDashboard() {
  const [activeTab, setActiveTab] = useState("assessment")
  const [patientData, setPatientData] = useState<EnhancedPatientData>(ENHANCED_SAMPLE_PATIENT)
  const [prediction, setPrediction] = useState<EnhancedPredictionResponse | null>(null)
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([])
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState<"online" | "offline" | "checking">("checking")
  const [error, setError] = useState<string | null>(null)
  const [batchResults, setBatchResults] = useState<any[]>([])

  // Check API health and load model info on component mount
  useEffect(() => {
    checkApiHealth()
    loadFeatureImportance()
    loadModelInfo()
  }, [])

  const checkApiHealth = async () => {
    try {
      setApiStatus("checking")
      const response = await axios.get(`${API_BASE_URL}/health`)
      setApiStatus("online")
      return response.data
    } catch (error) {
      setApiStatus("offline")
      return null
    }
  }

  const loadFeatureImportance = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/feature_importance`)
      setFeatureImportance(response.data.feature_importance || [])
    } catch (error) {
      console.log("Feature importance unavailable")
    }
  }

  const loadModelInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model_info`)
      setModelInfo(response.data)
    } catch (error) {
      console.log("Model info unavailable")
    }
  }

  const validateModel = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/validate_model`)
      return response.data
    } catch (error) {
      console.log("Model validation failed")
      return null
    }
  }

  const handleEnhancedPredict = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, patientData)
      setPrediction(response.data)
    } catch (error) {
      setError("Failed to get prediction. Please check your connection and try again.")
      console.error("Prediction error:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleBatchPredict = async () => {
    const batchPatients = [
      { ...patientData, patient_id: "batch_1" },
      { ...patientData, Age: 45, Lactate: 1.2, patient_id: "batch_2" },
      { ...patientData, Age: 75, WBC: 20, patient_id: "batch_3" }
    ]

    try {
      const response = await axios.post(`${API_BASE_URL}/predict_batch`, batchPatients)
      setBatchResults(response.data.predictions || [])
    } catch (error) {
      console.error("Batch prediction error:", error)
    }
  }

  const loadEnhancedSampleData = () => {
    setPatientData(ENHANCED_SAMPLE_PATIENT)
  }

  const handleInputChange = (field: keyof EnhancedPatientData, value: string) => {
    setPatientData((prev) => ({
      ...prev,
      [field]: field === 'Gender' ? parseInt(value) : (parseFloat(value) || undefined)
    }))
  }

  const getRiskBadge = (riskLevel: string) => {
    const config = ENHANCED_RISK_LEVELS[riskLevel as keyof typeof ENHANCED_RISK_LEVELS]
    return (
      <Badge className={`${config.color} text-white font-semibold px-4 py-2 text-sm shadow-lg`}>
        {riskLevel.toUpperCase()} RISK
      </Badge>
    )
  }

  const getAlertIcon = (alert: string) => {
    if (alert.includes("🔴")) return <XCircle className="h-4 w-4 text-red-500" />
    if (alert.includes("🟡")) return <AlertTriangle className="h-4 w-4 text-yellow-500" />
    if (alert.includes("🚨")) return <AlertTriangle className="h-4 w-4 text-red-600 animate-pulse" />
    return <Info className="h-4 w-4 text-blue-500" />
  }

  const getShapData = () => {
    if (!prediction?.shap_explanations) return []

    return Object.entries(prediction.shap_explanations)
      .map(([feature, data]) => ({
        feature: feature.replace(/_/g, " ").toUpperCase(),
        contribution: data.shap_contribution,
        absContribution: Math.abs(data.shap_contribution),
        impact: data.impact,
        value: data.value,
      }))
      .sort((a, b) => b.absContribution - a.absContribution)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Enhanced Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm shadow-lg">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl shadow-lg">
                <Stethoscope className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Enhanced Sepsis Prediction System
                </h1>
                <p className="text-sm text-gray-600">
                  {modelInfo?.model_version === "enhanced_preprocessing" ? "Enhanced" : "Standard"} XGBoost + SHAP Model
                  {modelInfo && ` • ${modelInfo.total_features} Features • API v${modelInfo.api_version}`}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-4 py-2 bg-gray-100 rounded-lg">
                <div
                  className={`w-3 h-3 rounded-full ${
                    apiStatus === "online"
                      ? "bg-green-500 animate-pulse"
                      : apiStatus === "offline"
                        ? "bg-red-500"
                        : "bg-yellow-500"
                  }`}
                />
                <span className="text-sm font-medium">
                  API {apiStatus === "checking" ? "Checking..." : apiStatus}
                </span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={checkApiHealth}
                className="border-blue-200 hover:bg-blue-50"
              >
                Refresh Status
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          <TabsList className="grid w-full grid-cols-4 bg-white/70 backdrop-blur-sm p-1 shadow-lg rounded-xl">
            <TabsTrigger
              value="assessment"
              className="data-[state=active]:bg-blue-500 data-[state=active]:text-white rounded-lg"
            >
              <Activity className="h-4 w-4 mr-2" />
              Assessment
            </TabsTrigger>
            <TabsTrigger
              value="insights"
              className="data-[state=active]:bg-indigo-500 data-[state=active]:text-white rounded-lg"
            >
              <Brain className="h-4 w-4 mr-2" />
              Model Insights
            </TabsTrigger>
            <TabsTrigger
              value="batch"
              className="data-[state=active]:bg-purple-500 data-[state=active]:text-white rounded-lg"
            >
              <Users className="h-4 w-4 mr-2" />
              Batch Analysis
            </TabsTrigger>
            <TabsTrigger
              value="dashboard"
              className="data-[state=active]:bg-green-500 data-[state=active]:text-white rounded-lg"
            >
              <TrendingUp className="h-4 w-4 mr-2" />
              Dashboard
            </TabsTrigger>
          </TabsList>

          {/* Enhanced Assessment Tab */}
          <TabsContent value="assessment" className="space-y-8">
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
              {/* Enhanced Patient Input Form */}
              <div className="xl:col-span-2">
                <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
                  <CardHeader className="bg-gradient-to-r from-blue-500/10 to-indigo-500/10 rounded-t-lg">
                    <CardTitle className="flex items-center gap-2 text-blue-700">
                      <Activity className="h-6 w-6" />
                      Enhanced Patient Assessment
                    </CardTitle>
                    <CardDescription>
                      Comprehensive clinical data input with {modelInfo?.total_features || "40+"} supported parameters
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-8 p-8">
                    {/* Demographics */}
                    <div className="space-y-4">
                      <h3 className="font-semibold text-lg text-gray-800 flex items-center gap-2">
                        <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                        Demographics
                      </h3>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="Age">Age (years) *</Label>
                          <Input
                            id="Age"
                            type="number"
                            value={patientData.Age}
                            onChange={(e) => handleInputChange("Age", e.target.value)}
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="Gender">Gender *</Label>
                          <Select
                            value={patientData.Gender.toString()}
                            onValueChange={(value) => handleInputChange("Gender", value)}
                          >
                            <SelectTrigger className="border-gray-300 focus:border-blue-500">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="0">Female</SelectItem>
                              <SelectItem value="1">Male</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="ICULOS">ICU Hours</Label>
                          <Input
                            id="ICULOS"
                            type="number"
                            value={patientData.ICULOS || ""}
                            onChange={(e) => handleInputChange("ICULOS", e.target.value)}
                            placeholder="24"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                      </div>
                    </div>

                    <Separator />

                    {/* Vital Signs */}
                    <div className="space-y-4">
                      <h3 className="font-semibold text-lg text-gray-800 flex items-center gap-2">
                        <Heart className="h-5 w-5 text-red-500" />
                        Vital Signs
                      </h3>
                      <div className="grid grid-cols-4 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="HR">Heart Rate (bpm)</Label>
                          <Input
                            id="HR"
                            type="number"
                            value={patientData.HR || ""}
                            onChange={(e) => handleInputChange("HR", e.target.value)}
                            placeholder="60-100"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="SBP">Systolic BP</Label>
                          <Input
                            id="SBP"
                            type="number"
                            value={patientData.SBP || ""}
                            onChange={(e) => handleInputChange("SBP", e.target.value)}
                            placeholder="90-140"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="DBP">Diastolic BP</Label>
                          <Input
                            id="DBP"
                            type="number"
                            value={patientData.DBP || ""}
                            onChange={(e) => handleInputChange("DBP", e.target.value)}
                            placeholder="60-90"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="MAP">MAP</Label>
                          <Input
                            id="MAP"
                            type="number"
                            value={patientData.MAP || ""}
                            onChange={(e) => handleInputChange("MAP", e.target.value)}
                            placeholder="70-100"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="Temp">Temperature (°C)</Label>
                          <Input
                            id="Temp"
                            type="number"
                            step="0.1"
                            value={patientData.Temp || ""}
                            onChange={(e) => handleInputChange("Temp", e.target.value)}
                            placeholder="36.1-37.2"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="Resp">Respiratory Rate</Label>
                          <Input
                            id="Resp"
                            type="number"
                            value={patientData.Resp || ""}
                            onChange={(e) => handleInputChange("Resp", e.target.value)}
                            placeholder="12-20"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="O2Sat">O2 Saturation (%)</Label>
                          <Input
                            id="O2Sat"
                            type="number"
                            value={patientData.O2Sat || ""}
                            onChange={(e) => handleInputChange("O2Sat", e.target.value)}
                            placeholder="95-100"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                      </div>
                    </div>

                    <Separator />

                    {/* Laboratory Values */}
                    <div className="space-y-4">
                      <h3 className="font-semibold text-lg text-gray-800 flex items-center gap-2">
                        <Droplets className="h-5 w-5 text-blue-500" />
                        Laboratory Values
                      </h3>
                      <div className="grid grid-cols-4 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="WBC">WBC (K/μL)</Label>
                          <Input
                            id="WBC"
                            type="number"
                            step="0.1"
                            value={patientData.WBC || ""}
                            onChange={(e) => handleInputChange("WBC", e.target.value)}
                            placeholder="4.0-11.0"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="Hgb">Hemoglobin</Label>
                          <Input
                            id="Hgb"
                            type="number"
                            step="0.1"
                            value={patientData.Hgb || ""}
                            onChange={(e) => handleInputChange("Hgb", e.target.value)}
                            placeholder="12-16"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="Platelets">Platelets</Label>
                          <Input
                            id="Platelets"
                            type="number"
                            value={patientData.Platelets || ""}
                            onChange={(e) => handleInputChange("Platelets", e.target.value)}
                            placeholder="150-450"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="Lactate">Lactate</Label>
                          <Input
                            id="Lactate"
                            type="number"
                            step="0.1"
                            value={patientData.Lactate || ""}
                            onChange={(e) => handleInputChange("Lactate", e.target.value)}
                            placeholder="0.5-2.2"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="Creatinine">Creatinine</Label>
                          <Input
                            id="Creatinine"
                            type="number"
                            step="0.1"
                            value={patientData.Creatinine || ""}
                            onChange={(e) => handleInputChange("Creatinine", e.target.value)}
                            placeholder="0.6-1.2"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="BUN">BUN</Label>
                          <Input
                            id="BUN"
                            type="number"
                            value={patientData.BUN || ""}
                            onChange={(e) => handleInputChange("BUN", e.target.value)}
                            placeholder="7-20"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="Glucose">Glucose</Label>
                          <Input
                            id="Glucose"
                            type="number"
                            value={patientData.Glucose || ""}
                            onChange={(e) => handleInputChange("Glucose", e.target.value)}
                            placeholder="70-100"
                            className="border-gray-300 focus:border-blue-500"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="flex gap-4 pt-6">
                      <Button
                        onClick={handleEnhancedPredict}
                        disabled={isLoading || apiStatus === "offline"}
                        className="flex-1 bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 shadow-lg text-white"
                        size="lg"
                      >
                        {isLoading ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                            Analyzing...
                          </>
                        ) : (
                          <>
                            <Brain className="h-4 w-4 mr-2" />
                            Predict Risk
                          </>
                        )}
                      </Button>
                      <Button
                        variant="outline"
                        onClick={loadEnhancedSampleData}
                        className="border-blue-200 text-blue-600 hover:bg-blue-50"
                        size="lg"
                      >
                        Load Sample
                      </Button>
                    </div>

                    {error && (
                      <Alert className="border-red-200 bg-red-50">
                        <AlertTriangle className="h-4 w-4 text-red-500" />
                        <AlertDescription className="text-red-700">{error}</AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Enhanced Prediction Results */}
              <div className="xl:col-span-1">
                <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm h-fit">
                  <CardHeader className="bg-gradient-to-r from-indigo-500/10 to-purple-500/10 rounded-t-lg">
                    <CardTitle className="flex items-center gap-2 text-indigo-700">
                      <Shield className="h-6 w-6" />
                      Risk Assessment
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-6">
                    {prediction ? (
                      <div className="space-y-6">
                        {/* Enhanced Risk Level Indicator */}
                        <div className={`p-6 rounded-xl ${ENHANCED_RISK_LEVELS[prediction.risk_level].bgColor} shadow-inner`}>
                          <div className="text-center space-y-4">
                            <div className="flex justify-center">{getRiskBadge(prediction.risk_level)}</div>
                            <div className="space-y-2">
                              <h3 className={`text-2xl font-bold ${ENHANCED_RISK_LEVELS[prediction.risk_level].textColor}`}>
                                Risk Score: {prediction.risk_score}/4
                              </h3>
                              <p className="text-sm text-gray-600">{prediction.prediction}</p>
                              <p className="text-xs text-gray-500">
                                Confidence: {(prediction.confidence * 100).toFixed(1)}% • 
                                Processing: {prediction.processing_time_ms.toFixed(1)}ms
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* Enhanced Probability Gauges */}
                        <div className="space-y-4">
                          <div className="space-y-2 p-4 bg-green-50 rounded-lg border border-green-200">
                            <div className="flex justify-between text-sm">
                              <span className="text-green-700 font-medium">Survival Probability</span>
                              <span className="font-bold text-green-700">
                                {(prediction.survival_probability * 100).toFixed(1)}%
                              </span>
                            </div>
                            <Progress value={prediction.survival_probability * 100} className="h-3" />
                          </div>
                          <div className="space-y-2 p-4 bg-red-50 rounded-lg border border-red-200">
                            <div className="flex justify-between text-sm">
                              <span className="text-red-700 font-medium">Mortality Risk</span>
                              <span className="font-bold text-red-700">
                                {(prediction.mortality_probability * 100).toFixed(1)}%
                              </span>
                            </div>
                            <Progress value={prediction.mortality_probability * 100} className="h-3" />
                          </div>
                        </div>

                        {/* Clinical Alerts */}
                        {prediction.clinical_alerts && prediction.clinical_alerts.length > 0 && (
                          <div className="space-y-3">
                            <h4 className="font-semibold text-gray-800">Clinical Alerts</h4>
                            <div className="space-y-2 max-h-32 overflow-y-auto">
                              {prediction.clinical_alerts.slice(0, 5).map((alert, index) => (
                                <div
                                  key={index}
                                  className="flex items-start gap-2 p-2 bg-gray-50 rounded-lg text-sm"
                                >
                                  {getAlertIcon(alert)}
                                  <span className="text-gray-700">{alert.replace(/🔴|🟡|🚨|⚠️/g, "").trim()}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Recommendations */}
                        {prediction.recommendations && prediction.recommendations.length > 0 && (
                          <div className="space-y-3">
                            <h4 className="font-semibold text-gray-800">Recommendations</h4>
                            <div className="space-y-2 max-h-32 overflow-y-auto">
                              {prediction.recommendations.slice(0, 3).map((rec, index) => (
                                <div
                                  key={index}
                                  className="flex items-start gap-2 p-2 bg-blue-50 rounded-lg text-sm"
                                >
                                  <CheckCircle className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                                  <span className="text-blue-700">{rec}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Model Info */}
                        <div className="text-xs text-gray-500 p-3 bg-gray-50 rounded-lg">
                          <div className="flex justify-between">
                            <span>Model: {prediction.model_version}</span>
                            <span>ID: {prediction.patient_id}</span>
                          </div>
                          <div className="mt-1">
                            Timestamp: {new Date(prediction.timestamp).toLocaleString()}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-12 text-gray-500">
                        <div className="p-4 bg-gray-100 rounded-full w-fit mx-auto mb-4">
                          <Shield className="h-12 w-12 opacity-50" />
                        </div>
                        <p>Enter patient data and click "Predict Risk" to see assessment results</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Enhanced Model Insights Tab */}
          <TabsContent value="insights" className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Feature Importance */}
              <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
                <CardHeader className="bg-gradient-to-r from-indigo-500/10 to-purple-500/10 rounded-t-lg">
                  <CardTitle className="text-indigo-700">Global Feature Importance</CardTitle>
                  <CardDescription>
                    Top clinical parameters impacting sepsis prediction
                    {apiStatus === "offline" && <span className="text-yellow-600 font-medium"> (Demo Data)</span>}
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-6">
                  {featureImportance.length > 0 ? (
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={featureImportance.slice(0, 10)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
                        <YAxis />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "white",
                            border: "1px solid #e2e8f0",
                            borderRadius: "8px",
                            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
                          }}
                        />
                        <Bar dataKey="importance" fill="url(#colorGradient)" radius={[4, 4, 0, 0]} />
                        <defs>
                          <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#3b82f6" />
                            <stop offset="100%" stopColor="#6366f1" />
                          </linearGradient>
                        </defs>
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center py-12 text-gray-500">
                      <Info className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Loading feature importance data...</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* SHAP Explanations */}
              <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
                <CardHeader className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-t-lg">
                  <CardTitle className="text-purple-700">SHAP Feature Contributions</CardTitle>
                  <CardDescription>
                    Individual feature impacts for current prediction
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-6">
                  {prediction?.shap_explanations ? (
                    <div className="space-y-3">
                      {getShapData().slice(0, 8).map((item, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-3 bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg border"
                        >
                          <div className="flex-1">
                            <span className="text-sm font-medium">{item.feature}</span>
                            <div className="text-xs text-gray-500">Value: {item.value}</div>
                          </div>
                          <div className="text-right">
                            <div className={`text-sm font-semibold ${
                              item.contribution > 0 ? "text-red-600" : "text-green-600"
                            }`}>
                              {item.contribution > 0 ? "+" : ""}{item.contribution.toFixed(3)}
                            </div>
                            <div className={`text-xs font-medium ${
                              item.impact === "increases_survival" ? "text-green-600" : "text-red-600"
                            }`}>
                              {item.impact.replace("_", " ")}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-12 text-gray-500">
                      <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Run a prediction to see SHAP explanations</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Batch Analysis Tab */}
          <TabsContent value="batch" className="space-y-8">
            <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-t-lg">
                <CardTitle className="flex items-center gap-2 text-purple-700">
                  <Users className="h-6 w-6" />
                  Batch Patient Analysis
                </CardTitle>
                <CardDescription>
                  Analyze multiple patients simultaneously for population-level insights
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6">
                <div className="space-y-6">
                  <Button
                    onClick={handleBatchPredict}
                    disabled={apiStatus === "offline"}
                    className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white"
                  >
                    <Users className="h-4 w-4 mr-2" />
                    Run Batch Analysis (3 Patients)
                  </Button>

                  {batchResults.length > 0 && (
                    <div className="space-y-4">
                      <h3 className="font-semibold text-lg">Batch Results</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {batchResults.map((result, index) => (
                          <Card key={index} className="border border-gray-200">
                            <CardContent className="p-4">
                              <div className="space-y-2">
                                <div className="flex justify-between items-center">
                                  <span className="font-medium">Patient {index + 1}</span>
                                  {result.error ? (
                                    <Badge className="bg-red-100 text-red-700">Error</Badge>
                                  ) : (
                                    getRiskBadge(result.risk_level)
                                  )}
                                </div>
                                {!result.error && (
                                  <>
                                    <div className="text-sm text-gray-600">
                                      Mortality Risk: {(result.mortality_probability * 100).toFixed(1)}%
                                    </div>
                                    <div className="text-sm text-gray-600">
                                      Risk Score: {result.risk_score}/4
                                    </div>
                                  </>
                                )}
                                {result.error && (
                                  <div className="text-sm text-red-600">{result.error}</div>
                                )}
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Enhanced Dashboard Tab */}
          <TabsContent value="dashboard" className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* System Status */}
              <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg text-green-700">System Status</CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <div className="flex items-center gap-3">
                    {apiStatus === "online" ? (
                      <div className="p-2 bg-green-100 rounded-lg">
                        <CheckCircle className="h-8 w-8 text-green-600" />
                      </div>
                    ) : (
                      <div className="p-2 bg-red-100 rounded-lg">
                        <XCircle className="h-8 w-8 text-red-600" />
                      </div>
                    )}
                    <div>
                      <p className="font-semibold">API Status</p>
                      <p className="text-sm text-gray-600 capitalize">{apiStatus}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Model Info */}
              <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg text-blue-700">Model Info</CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <div className="space-y-2">
                    <div className="text-sm">
                      <span className="font-medium">Type:</span> {modelInfo?.model_type || "XGBoost"}
                    </div>
                    <div className="text-sm">
                      <span className="font-medium">Version:</span> {modelInfo?.model_version || "Standard"}
                    </div>
                    <div className="text-sm">
                      <span className="font-medium">Features:</span> {modelInfo?.total_features || "N/A"}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Performance Metrics */}
              <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg text-purple-700">Performance</CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <div className="space-y-2">
                    <div className="text-sm">
                      <span className="font-medium">SHAP:</span> {modelInfo?.has_shap ? "✅" : "❌"}
                    </div>
                    <div className="text-sm">
                      <span className="font-medium">Enhanced:</span> {modelInfo?.preprocessing_info?.enhanced_preprocessing ? "✅" : "❌"}
                    </div>
                    <div className="text-sm">
                      <span className="font-medium">API:</span> v{modelInfo?.api_version || "2.0"}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Quick Actions */}
              <Card className="shadow-lg border-0 bg-white/90 backdrop-blur-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg text-indigo-700">Quick Actions</CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <div className="space-y-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={validateModel}
                      className="w-full text-xs"
                    >
                      Validate Model
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={loadModelInfo}
                      className="w-full text-xs"
                    >
                      Refresh Info
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Enhanced Features Overview */}
            {modelInfo?.preprocessing_info?.enhanced_preprocessing && (
              <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm">
                <CardHeader className="bg-gradient-to-r from-green-500/10 to-blue-500/10 rounded-t-lg">
                  <CardTitle className="text-green-700">Enhanced Preprocessing Features</CardTitle>
                  <CardDescription>
                    Advanced clinical feature engineering and model capabilities
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="space-y-3">
                      <h4 className="font-semibold text-gray-800">Clinical Scores</h4>
                      <div className="space-y-1">
                        {modelInfo.supported_features?.clinical_scores?.map((score, index) => (
                          <div key={index} className="text-sm text-gray-600 flex items-center gap-2">
                            <CheckCircle className="h-3 w-3 text-green-500" />
                            {score}
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="space-y-3">
                      <h4 className="font-semibold text-gray-800">Physiological Indices</h4>
                      <div className="space-y-1">
                        {modelInfo.supported_features?.physiological_indices?.map((index, idx) => (
                          <div key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                            <CheckCircle className="h-3 w-3 text-blue-500" />
                            {index}
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="space-y-3">
                      <h4 className="font-semibold text-gray-800">Advanced Features</h4>
                      <div className="space-y-1">
                        <div className="text-sm text-gray-600 flex items-center gap-2">
                          <CheckCircle className="h-3 w-3 text-purple-500" />
                          Clinical Alerts
                        </div>
                        <div className="text-sm text-gray-600 flex items-center gap-2">
                          <CheckCircle className="h-3 w-3 text-purple-500" />
                          Batch Processing
                        </div>
                        <div className="text-sm text-gray-600 flex items-center gap-2">
                          <CheckCircle className="h-3 w-3 text-purple-500" />
                          SHAP Explanations
                        </div>
                        <div className="text-sm text-gray-600 flex items-center gap-2">
                          <CheckCircle className="h-3 w-3 text-purple-500" />
                          Temporal Features
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}