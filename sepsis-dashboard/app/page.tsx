"use client"

import EnhancedSepsisDashboard from "@/components/EnhancedSepsisDashboard"

export default function Home() {
  return <EnhancedSepsisDashboard />
}
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
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { Activity, Heart, Droplets, AlertTriangle, CheckCircle, XCircle, Info, Stethoscope } from "lucide-react"
import axios from "axios"

// API Configuration
const API_BASE_URL = "http://localhost:8000"

// Sample patient data
const SAMPLE_PATIENT = {
  age: 65,
  gender: 1,
  heart_rate: 95,
  systolic_bp: 110,
  temperature: 38.2,
  respiratory_rate: 22,
  wbc_count: 12.5,
  lactate: 3.2,
  sofa_score: 4,
}

// Risk level configurations
const RISK_LEVELS = {
  low: {
    color: "bg-success text-white",
    textColor: "text-success",
    bgColor: "bg-success/10 border border-success/20",
    iconColor: "text-success",
  },
  moderate: {
    color: "bg-warning text-warning-foreground",
    textColor: "text-warning",
    bgColor: "bg-warning/10 border border-warning/20",
    iconColor: "text-warning",
  },
  high: {
    color: "bg-orange-500 text-white",
    textColor: "text-orange-600",
    bgColor: "bg-orange-50 border border-orange-200",
    iconColor: "text-orange-500",
  },
  critical: {
    color: "bg-destructive text-destructive-foreground",
    textColor: "text-destructive",
    bgColor: "bg-destructive/10 border border-destructive/20",
    iconColor: "text-destructive",
  },
}

interface PatientData {
  age: number
  gender: number
  heart_rate: number
  systolic_bp: number
  temperature: number
  respiratory_rate: number
  wbc_count: number
  lactate: number
  sofa_score: number
}

interface PredictionResult {
  survival_probability: number
  mortality_probability: number
  risk_level: "low" | "moderate" | "high" | "critical"
  prediction: string
  shap_explanations?: Record<
    string,
    {
      value: number
      shap_contribution: number
      impact: string
    }
  >
}

interface FeatureImportance {
  feature: string
  importance: number
  description: string
}

export default function SepsisDashboard() {
  const [activeTab, setActiveTab] = useState("assessment")
  const [patientData, setPatientData] = useState<PatientData>(SAMPLE_PATIENT)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState<"online" | "offline" | "checking">("checking")
  const [error, setError] = useState<string | null>(null)

  // Mock feature importance data for when API is unavailable
  const MOCK_FEATURE_IMPORTANCE: FeatureImportance[] = [
    { feature: "SOFA Score", importance: 0.28, description: "Sequential Organ Failure Assessment" },
    { feature: "Lactate", importance: 0.22, description: "Blood lactate levels" },
    { feature: "Age", importance: 0.18, description: "Patient age in years" },
    { feature: "WBC Count", importance: 0.12, description: "White blood cell count" },
    { feature: "Heart Rate", importance: 0.08, description: "Heart rate in beats per minute" },
    { feature: "Temperature", importance: 0.06, description: "Body temperature" },
    { feature: "Systolic BP", importance: 0.04, description: "Systolic blood pressure" },
    { feature: "Respiratory Rate", importance: 0.02, description: "Respiratory rate per minute" },
  ]

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth()
    loadFeatureImportance()
  }, [])

  const checkApiHealth = async () => {
    try {
      setApiStatus("checking")
      await axios.get(`${API_BASE_URL}/health`)
      setApiStatus("online")
    } catch (error) {
      setApiStatus("offline")
    }
  }

  const loadFeatureImportance = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/feature_importance`)
      setFeatureImportance(response.data)
    } catch (error) {
      console.log("API unavailable, using mock feature importance data")
      setFeatureImportance(MOCK_FEATURE_IMPORTANCE)
    }
  }

  const handlePredict = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, patientData)
      setPrediction(response.data)
    } catch (error) {
      if (apiStatus === "offline") {
        const mockPrediction: PredictionResult = {
          survival_probability: 0.72,
          mortality_probability: 0.28,
          risk_level: "moderate",
          prediction: "Moderate risk patient - monitor closely",
          shap_explanations: {
            sofa_score: { value: patientData.sofa_score, shap_contribution: 0.15, impact: "increases_mortality" },
            lactate: { value: patientData.lactate, shap_contribution: 0.12, impact: "increases_mortality" },
            age: { value: patientData.age, shap_contribution: 0.08, impact: "increases_mortality" },
            wbc_count: { value: patientData.wbc_count, shap_contribution: 0.06, impact: "increases_mortality" },
            heart_rate: { value: patientData.heart_rate, shap_contribution: -0.03, impact: "increases_survival" },
          },
        }
        setPrediction(mockPrediction)
        setError("Using demo prediction - API server not available")
      } else {
        setError("Failed to get prediction. Please check your connection and try again.")
      }
      console.error("Prediction error:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadSampleData = () => {
    setPatientData(SAMPLE_PATIENT)
  }

  const handleInputChange = (field: keyof PatientData, value: string) => {
    setPatientData((prev) => ({
      ...prev,
      [field]: Number.parseFloat(value) || 0,
    }))
  }

  const getRiskBadge = (riskLevel: string) => {
    const config = RISK_LEVELS[riskLevel as keyof typeof RISK_LEVELS]
    return <Badge className={`${config.color} text-white font-semibold px-3 py-1`}>{riskLevel.toUpperCase()}</Badge>
  }

  const getShapData = () => {
    if (!prediction?.shap_explanations) return []

    return Object.entries(prediction.shap_explanations)
      .map(([feature, data]) => ({
        feature: feature.replace("_", " ").toUpperCase(),
        contribution: Math.abs(data.shap_contribution),
        impact: data.impact,
        value: data.value,
      }))
      .sort((a, b) => b.contribution - a.contribution)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      {/* Header */}
      <header className="border-b bg-card/80 backdrop-blur-sm shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-xl">
                <Stethoscope className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  Sepsis Prediction System
                </h1>
                <p className="text-sm text-muted-foreground">XGBoost + SHAP Model</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-lg">
                <div
                  className={`w-2 h-2 rounded-full ${
                    apiStatus === "online"
                      ? "bg-success animate-pulse"
                      : apiStatus === "offline"
                        ? "bg-destructive"
                        : "bg-warning"
                  }`}
                />
                <span className="text-sm font-medium">API {apiStatus === "checking" ? "Checking..." : apiStatus}</span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={checkApiHealth}
                className="border-primary/20 hover:bg-primary/5 bg-transparent"
              >
                Refresh Status
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 bg-card/50 backdrop-blur-sm p-1">
            <TabsTrigger
              value="assessment"
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              Patient Assessment
            </TabsTrigger>
            <TabsTrigger
              value="insights"
              className="data-[state=active]:bg-secondary data-[state=active]:text-secondary-foreground"
            >
              Model Insights
            </TabsTrigger>
            <TabsTrigger
              value="dashboard"
              className="data-[state=active]:bg-accent data-[state=active]:text-accent-foreground"
            >
              Dashboard
            </TabsTrigger>
          </TabsList>

          {/* Patient Assessment Tab */}
          <TabsContent value="assessment" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Patient Input Form */}
              <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                <CardHeader className="bg-gradient-to-r from-primary/5 to-secondary/5 rounded-t-lg">
                  <CardTitle className="flex items-center gap-2 text-primary">
                    <Activity className="h-5 w-5" />
                    Patient Information
                  </CardTitle>
                  <CardDescription>
                    Enter patient vital signs and clinical data for sepsis risk assessment
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6 p-6">
                  {/* Demographics */}
                  <div className="space-y-4">
                    <h3 className="font-semibold text-foreground flex items-center gap-2">
                      <div className="w-2 h-2 bg-info rounded-full"></div>
                      Demographics
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="age">Age (years)</Label>
                        <Input
                          id="age"
                          type="number"
                          value={patientData.age}
                          onChange={(e) => handleInputChange("age", e.target.value)}
                          placeholder="65"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="gender">Gender</Label>
                        <Select
                          value={patientData.gender.toString()}
                          onValueChange={(value) => handleInputChange("gender", value)}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="0">Female</SelectItem>
                            <SelectItem value="1">Male</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>

                  <Separator className="bg-gradient-to-r from-transparent via-border to-transparent" />

                  {/* Vital Signs */}
                  <div className="space-y-4">
                    <h3 className="font-semibold text-foreground flex items-center gap-2">
                      <Heart className="h-4 w-4 text-destructive" />
                      Vital Signs
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="heart_rate">Heart Rate (bpm)</Label>
                        <Input
                          id="heart_rate"
                          type="number"
                          value={patientData.heart_rate}
                          onChange={(e) => handleInputChange("heart_rate", e.target.value)}
                          placeholder="60-100"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="systolic_bp">Systolic BP (mmHg)</Label>
                        <Input
                          id="systolic_bp"
                          type="number"
                          value={patientData.systolic_bp}
                          onChange={(e) => handleInputChange("systolic_bp", e.target.value)}
                          placeholder="90-140"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="temperature">Temperature (°C)</Label>
                        <Input
                          id="temperature"
                          type="number"
                          step="0.1"
                          value={patientData.temperature}
                          onChange={(e) => handleInputChange("temperature", e.target.value)}
                          placeholder="36.1-37.2"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="respiratory_rate">Respiratory Rate (/min)</Label>
                        <Input
                          id="respiratory_rate"
                          type="number"
                          value={patientData.respiratory_rate}
                          onChange={(e) => handleInputChange("respiratory_rate", e.target.value)}
                          placeholder="12-20"
                        />
                      </div>
                    </div>
                  </div>

                  <Separator className="bg-gradient-to-r from-transparent via-border to-transparent" />

                  {/* Lab Values */}
                  <div className="space-y-4">
                    <h3 className="font-semibold text-foreground flex items-center gap-2">
                      <Droplets className="h-4 w-4 text-secondary" />
                      Laboratory Values
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="wbc_count">WBC Count (×10³/μL)</Label>
                        <Input
                          id="wbc_count"
                          type="number"
                          step="0.1"
                          value={patientData.wbc_count}
                          onChange={(e) => handleInputChange("wbc_count", e.target.value)}
                          placeholder="4.0-11.0"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="lactate">Lactate (mmol/L)</Label>
                        <Input
                          id="lactate"
                          type="number"
                          step="0.1"
                          value={patientData.lactate}
                          onChange={(e) => handleInputChange("lactate", e.target.value)}
                          placeholder="0.5-2.2"
                        />
                      </div>
                    </div>
                  </div>

                  <Separator className="bg-gradient-to-r from-transparent via-border to-transparent" />

                  {/* Clinical Scores */}
                  <div className="space-y-4">
                    <h3 className="font-semibold text-foreground flex items-center gap-2">
                      <div className="w-2 h-2 bg-accent rounded-full"></div>
                      Clinical Scores
                    </h3>
                    <div className="space-y-2">
                      <Label htmlFor="sofa_score">SOFA Score</Label>
                      <Input
                        id="sofa_score"
                        type="number"
                        value={patientData.sofa_score}
                        onChange={(e) => handleInputChange("sofa_score", e.target.value)}
                        placeholder="0-24"
                      />
                    </div>
                  </div>

                  <div className="flex gap-3 pt-4">
                    <Button
                      onClick={handlePredict}
                      disabled={isLoading || apiStatus === "offline"}
                      className="flex-1 bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 shadow-lg"
                    >
                      {isLoading ? "Analyzing..." : "Predict Risk"}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={loadSampleData}
                      className="border-secondary text-secondary hover:bg-secondary/10 bg-transparent"
                    >
                      Load Sample
                    </Button>
                  </div>

                  {error && (
                    <Alert className="border-destructive/20 bg-destructive/5">
                      <AlertTriangle className="h-4 w-4 text-destructive" />
                      <AlertDescription className="text-destructive">{error}</AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>

              {/* Prediction Results */}
              <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                <CardHeader className="bg-gradient-to-r from-accent/5 to-primary/5 rounded-t-lg">
                  <CardTitle className="flex items-center gap-2 text-accent">
                    <AlertTriangle className="h-5 w-5" />
                    Risk Assessment Results
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-6">
                  {prediction ? (
                    <div className="space-y-6">
                      {/* Risk Level Indicator */}
                      <div className={`p-6 rounded-xl ${RISK_LEVELS[prediction.risk_level].bgColor} shadow-inner`}>
                        <div className="text-center space-y-3">
                          <div className="flex justify-center">{getRiskBadge(prediction.risk_level)}</div>
                          <h3 className={`text-2xl font-bold ${RISK_LEVELS[prediction.risk_level].textColor}`}>
                            {prediction.risk_level.toUpperCase()} RISK
                          </h3>
                          <p className="text-sm text-muted-foreground">Prediction: {prediction.prediction}</p>
                        </div>
                      </div>

                      {/* Probability Gauges */}
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2 p-4 bg-success/5 rounded-lg border border-success/10">
                          <div className="flex justify-between text-sm">
                            <span className="text-success font-medium">Survival Probability</span>
                            <span className="font-bold text-success">
                              {(prediction.survival_probability * 100).toFixed(1)}%
                            </span>
                          </div>
                          <Progress value={prediction.survival_probability * 100} className="h-3 bg-success/20" />
                        </div>
                        <div className="space-y-2 p-4 bg-destructive/5 rounded-lg border border-destructive/10">
                          <div className="flex justify-between text-sm">
                            <span className="text-destructive font-medium">Mortality Risk</span>
                            <span className="font-bold text-destructive">
                              {(prediction.mortality_probability * 100).toFixed(1)}%
                            </span>
                          </div>
                          <Progress value={prediction.mortality_probability * 100} className="h-3 bg-destructive/20" />
                        </div>
                      </div>

                      {/* SHAP Explanations */}
                      {prediction.shap_explanations && (
                        <div className="space-y-4">
                          <h4 className="font-semibold flex items-center gap-2">
                            <div className="w-2 h-2 bg-info rounded-full"></div>
                            Feature Contributions
                          </h4>
                          <div className="space-y-2">
                            {getShapData()
                              .slice(0, 5)
                              .map((item, index) => (
                                <div
                                  key={index}
                                  className="flex items-center justify-between p-3 bg-gradient-to-r from-muted/50 to-muted/30 rounded-lg border border-border/50"
                                >
                                  <div className="flex-1">
                                    <span className="text-sm font-medium">{item.feature}</span>
                                    <div className="text-xs text-muted-foreground">Value: {item.value}</div>
                                  </div>
                                  <div className="text-right">
                                    <div className="text-sm font-semibold">
                                      {item.contribution > 0 ? "+" : ""}
                                      {item.contribution.toFixed(3)}
                                    </div>
                                    <div
                                      className={`text-xs font-medium ${item.impact === "increases_survival" ? "text-success" : "text-destructive"}`}
                                    >
                                      {item.impact.replace("_", " ")}
                                    </div>
                                  </div>
                                </div>
                              ))}
                          </div>
                        </div>
                      )}

                      {/* Clinical Recommendations */}
                      <Alert className="border-info/20 bg-info/5">
                        <Info className="h-4 w-4 text-info" />
                        <AlertDescription className="text-info-foreground">
                          <strong>Clinical Note:</strong> This prediction is for clinical decision support only. Always
                          consider the complete clinical picture and follow institutional protocols.
                        </AlertDescription>
                      </Alert>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">
                      <div className="p-4 bg-muted/30 rounded-full w-fit mx-auto mb-4">
                        <AlertTriangle className="h-12 w-12 opacity-50" />
                      </div>
                      <p>Enter patient data and click "Predict Risk" to see assessment results</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Model Insights Tab */}
          <TabsContent value="insights" className="space-y-6">
            <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-secondary/5 to-accent/5 rounded-t-lg">
                <CardTitle className="text-secondary">Global Feature Importance</CardTitle>
                <CardDescription>
                  Understanding which clinical parameters have the most impact on sepsis prediction
                  {apiStatus === "offline" && <span className="text-warning font-medium"> (Demo Data)</span>}
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6">
                {featureImportance.length > 0 ? (
                  <div className="space-y-4">
                    {apiStatus === "offline" && (
                      <Alert className="border-warning/20 bg-warning/5">
                        <Info className="h-4 w-4 text-warning" />
                        <AlertDescription className="text-warning-foreground">
                          <strong>Demo Mode:</strong> API server not available. Showing sample feature importance data.
                        </AlertDescription>
                      </Alert>
                    )}
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={featureImportance}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
                        <YAxis />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--card))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: "8px",
                          }}
                        />
                        <Bar dataKey="importance" fill="url(#colorGradient)" radius={[4, 4, 0, 0]} />
                        <defs>
                          <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="hsl(var(--primary))" />
                            <stop offset="100%" stopColor="hsl(var(--secondary))" />
                          </linearGradient>
                        </defs>
                      </BarChart>
                    </ResponsiveContainer>
                    <div className="flex gap-3">
                      <Button
                        variant="outline"
                        onClick={loadFeatureImportance}
                        className="border-secondary text-secondary hover:bg-secondary/10 bg-transparent"
                      >
                        Refresh Data
                      </Button>
                      <Button
                        variant="outline"
                        onClick={checkApiHealth}
                        className="border-primary text-primary hover:bg-primary/10 bg-transparent"
                      >
                        Check API Status
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 text-muted-foreground">
                    <div className="p-4 bg-info/10 rounded-full w-fit mx-auto mb-4">
                      <Info className="h-12 w-12 text-info" />
                    </div>
                    <p>Loading feature importance data...</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Dashboard Tab */}
          <TabsContent value="dashboard" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                <CardHeader className="pb-3 bg-gradient-to-r from-success/5 to-success/10 rounded-t-lg">
                  <CardTitle className="text-lg text-success">System Status</CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <div className="flex items-center gap-3">
                    {apiStatus === "online" ? (
                      <div className="p-2 bg-success/10 rounded-lg">
                        <CheckCircle className="h-8 w-8 text-success" />
                      </div>
                    ) : apiStatus === "offline" ? (
                      <div className="p-2 bg-destructive/10 rounded-lg">
                        <XCircle className="h-8 w-8 text-destructive" />
                      </div>
                    ) : (
                      <div className="p-2 bg-warning/10 rounded-lg">
                        <AlertTriangle className="h-8 w-8 text-warning" />
                      </div>
                    )}
                    <div>
                      <p className="font-semibold">API Status</p>
                      <p className="text-sm text-muted-foreground capitalize">{apiStatus}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                <CardHeader className="pb-3 bg-gradient-to-r from-info/5 to-info/10 rounded-t-lg">
                  <CardTitle className="text-lg text-info">Model Info</CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Algorithm:</span>
                      <span className="text-sm font-semibold text-primary">XGBoost</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Explainability:</span>
                      <span className="text-sm font-semibold text-secondary">SHAP</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Features:</span>
                      <span className="text-sm font-semibold text-accent">9 Clinical</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
                <CardHeader className="pb-3 bg-gradient-to-r from-accent/5 to-accent/10 rounded-t-lg">
                  <CardTitle className="text-lg text-accent">Last Prediction</CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  {prediction ? (
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Risk Level:</span>
                        {getRiskBadge(prediction.risk_level)}
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Survival:</span>
                        <span className="text-sm font-semibold text-success">
                          {(prediction.survival_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No predictions yet</p>
                  )}
                </CardContent>
              </Card>
            </div>

            <Card className="shadow-lg border-0 bg-card/80 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-primary/5 via-secondary/5 to-accent/5 rounded-t-lg">
                <CardTitle className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  Clinical Guidelines
                </CardTitle>
                <CardDescription>Sepsis prediction and management guidelines</CardDescription>
              </CardHeader>
              <CardContent className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4 p-4 bg-success/5 rounded-lg border border-success/10">
                    <h4 className="font-semibold text-success flex items-center gap-2">
                      <div className="w-3 h-3 bg-success rounded-full"></div>
                      Low Risk (Green)
                    </h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Continue standard monitoring</li>
                      <li>• Routine vital signs assessment</li>
                      <li>• Follow standard care protocols</li>
                    </ul>
                  </div>
                  <div className="space-y-4 p-4 bg-warning/5 rounded-lg border border-warning/10">
                    <h4 className="font-semibold text-warning flex items-center gap-2">
                      <div className="w-3 h-3 bg-warning rounded-full"></div>
                      Moderate Risk (Yellow)
                    </h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Increase monitoring frequency</li>
                      <li>• Consider additional lab work</li>
                      <li>• Alert attending physician</li>
                    </ul>
                  </div>
                  <div className="space-y-4 p-4 bg-orange-50 rounded-lg border border-orange-200">
                    <h4 className="font-semibold text-orange-700 flex items-center gap-2">
                      <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                      High Risk (Orange)
                    </h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Immediate physician notification</li>
                      <li>• Consider ICU consultation</li>
                      <li>• Prepare for rapid response</li>
                    </ul>
                  </div>
                  <div className="space-y-4 p-4 bg-destructive/5 rounded-lg border border-destructive/10">
                    <h4 className="font-semibold text-destructive flex items-center gap-2">
                      <div className="w-3 h-3 bg-destructive rounded-full"></div>
                      Critical Risk (Red)
                    </h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Activate rapid response team</li>
                      <li>• Immediate ICU transfer</li>
                      <li>• Begin sepsis protocol</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
