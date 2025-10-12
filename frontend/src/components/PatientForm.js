import React, { useState } from 'react';
import { User, Heart, Thermometer, Activity, Droplets, Stethoscope, Brain, Clock } from 'lucide-react';

const PatientForm = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    // Demographics (Required)
    Age: '',
    Gender: '',
    
    // Core Vital Signs
    HR: '',
    O2Sat: '',
    Temp: '',
    SBP: '',
    MAP: '',
    DBP: '',
    Resp: '',
    
    // Laboratory Values
    Glucose: '',
    BUN: '',
    Creatinine: '',
    WBC: '',
    Hct: '',
    Hgb: '',
    Platelets: '',
    Lactate: '',
    
    // Additional Lab Values
    Bilirubin_total: '',
    AST: '',
    Alkalinephos: '',
    Calcium: '',
    Chloride: '',
    Magnesium: '',
    Phosphate: '',
    Potassium: '',
    
    // Blood Gas and Respiratory
    pH: '',
    PaCO2: '',
    BaseExcess: '',
    HCO3: '',
    FiO2: '',
    
    // Temporal
    ICULOS: '',
    
    // Clinical Context
    patient_id: '',
    unit: ''
  });

  const [errors, setErrors] = useState({});

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    // Required fields validation
    if (!formData.Age || formData.Age < 0 || formData.Age > 120) {
      newErrors.Age = 'Age must be between 0 and 120';
    }
    if (formData.Gender === '') {
      newErrors.Gender = 'Gender is required';
    }
    
    // Optional field validations (only if provided)
    if (formData.HR && (formData.HR < 20 || formData.HR > 250)) {
      newErrors.HR = 'Heart rate must be between 20 and 250 bpm';
    }
    if (formData.SBP && (formData.SBP < 50 || formData.SBP > 300)) {
      newErrors.SBP = 'Systolic BP must be between 50 and 300 mmHg';
    }
    if (formData.Temp && (formData.Temp < 30 || formData.Temp > 45)) {
      newErrors.Temp = 'Temperature must be between 30 and 45°C';
    }
    if (formData.Resp && (formData.Resp < 5 || formData.Resp > 60)) {
      newErrors.Resp = 'Respiratory rate must be between 5 and 60 breaths/min';
    }
    if (formData.O2Sat && (formData.O2Sat < 50 || formData.O2Sat > 100)) {
      newErrors.O2Sat = 'O2 Saturation must be between 50 and 100%';
    }
    if (formData.WBC && (formData.WBC < 0.1 || formData.WBC > 100)) {
      newErrors.WBC = 'WBC count must be between 0.1 and 100 K/μL';
    }
    if (formData.Lactate && (formData.Lactate < 0.1 || formData.Lactate > 30)) {
      newErrors.Lactate = 'Lactate must be between 0.1 and 30 mmol/L';
    }
    if (formData.pH && (formData.pH < 6.8 || formData.pH > 7.8)) {
      newErrors.pH = 'pH must be between 6.8 and 7.8';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      // Convert string values to numbers, only include non-empty values
      const processedData = {};
      
      Object.keys(formData).forEach(key => {
        const value = formData[key];
        if (value !== '') {
          if (key === 'Gender') {
            processedData[key] = parseInt(value);
          } else if (key === 'patient_id' || key === 'unit') {
            processedData[key] = value;
          } else {
            processedData[key] = parseFloat(value);
          }
        }
      });
      
      onSubmit(processedData);
    }
  };

  const loadSampleData = () => {
    setFormData({
      Age: '68',
      Gender: '1',
      HR: '105',
      O2Sat: '92',
      Temp: '38.7',
      SBP: '88',
      MAP: '62',
      DBP: '55',
      Resp: '26',
      Glucose: '145',
      BUN: '28',
      Creatinine: '1.8',
      WBC: '16.2',
      Hct: '32',
      Hgb: '10.5',
      Platelets: '95',
      Lactate: '3.8',
      pH: '7.32',
      HCO3: '18',
      ICULOS: '18',
      patient_id: 'demo_patient_001',
      unit: 'ICU'
    });
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800 flex items-center">
          <User className="mr-2 text-primary-600" />
          Patient Information
        </h2>
        <button
          type="button"
          onClick={loadSampleData}
          className="btn-secondary text-sm"
        >
          Load Sample Data
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
          {/* Demographics */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-700 border-b pb-2 flex items-center">
              <User className="mr-2 text-blue-500" size={20} />
              Demographics *
            </h3>
            
            <div>
              <label className="label">Age (years) *</label>
              <input
                type="number"
                name="Age"
                value={formData.Age}
                onChange={handleChange}
                className={`input-field ${errors.Age ? 'border-red-500' : ''}`}
                placeholder="Enter age"
                min="0"
                max="120"
                required
              />
              {errors.Age && <p className="text-red-500 text-sm mt-1">{errors.Age}</p>}
            </div>

            <div>
              <label className="label">Gender *</label>
              <select
                name="Gender"
                value={formData.Gender}
                onChange={handleChange}
                className={`input-field ${errors.Gender ? 'border-red-500' : ''}`}
                required
              >
                <option value="">Select gender</option>
                <option value="0">Female</option>
                <option value="1">Male</option>
              </select>
              {errors.Gender && <p className="text-red-500 text-sm mt-1">{errors.Gender}</p>}
            </div>

            <div>
              <label className="label">ICU Hours</label>
              <input
                type="number"
                name="ICULOS"
                value={formData.ICULOS}
                onChange={handleChange}
                className="input-field"
                placeholder="Hours in ICU"
                min="0"
              />
            </div>

            <div>
              <label className="label">Patient ID</label>
              <input
                type="text"
                name="patient_id"
                value={formData.patient_id}
                onChange={handleChange}
                className="input-field"
                placeholder="Patient identifier"
              />
            </div>

            <div>
              <label className="label">Unit</label>
              <input
                type="text"
                name="unit"
                value={formData.unit}
                onChange={handleChange}
                className="input-field"
                placeholder="Hospital unit (e.g., ICU)"
              />
            </div>
          </div>

          {/* Vital Signs */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-700 border-b pb-2 flex items-center">
              <Heart className="mr-2 text-red-500" size={20} />
              Vital Signs
            </h3>
            
            <div>
              <label className="label">Heart Rate (bpm)</label>
              <input
                type="number"
                name="HR"
                value={formData.HR}
                onChange={handleChange}
                className={`input-field ${errors.HR ? 'border-red-500' : ''}`}
                placeholder="60-100"
                min="20"
                max="250"
              />
              {errors.HR && <p className="text-red-500 text-sm mt-1">{errors.HR}</p>}
            </div>

            <div>
              <label className="label">Systolic BP (mmHg)</label>
              <input
                type="number"
                name="SBP"
                value={formData.SBP}
                onChange={handleChange}
                className={`input-field ${errors.SBP ? 'border-red-500' : ''}`}
                placeholder="90-140"
                min="50"
                max="300"
              />
              {errors.SBP && <p className="text-red-500 text-sm mt-1">{errors.SBP}</p>}
            </div>

            <div>
              <label className="label">Diastolic BP (mmHg)</label>
              <input
                type="number"
                name="DBP"
                value={formData.DBP}
                onChange={handleChange}
                className="input-field"
                placeholder="60-90"
                min="20"
                max="150"
              />
            </div>

            <div>
              <label className="label">MAP (mmHg)</label>
              <input
                type="number"
                name="MAP"
                value={formData.MAP}
                onChange={handleChange}
                className="input-field"
                placeholder="70-100"
                min="30"
                max="200"
              />
            </div>

            <div>
              <label className="label flex items-center">
                <Thermometer className="mr-1 text-orange-500" size={16} />
                Temperature (°C)
              </label>
              <input
                type="number"
                step="0.1"
                name="Temp"
                value={formData.Temp}
                onChange={handleChange}
                className={`input-field ${errors.Temp ? 'border-red-500' : ''}`}
                placeholder="36.1-37.2"
                min="30"
                max="45"
              />
              {errors.Temp && <p className="text-red-500 text-sm mt-1">{errors.Temp}</p>}
            </div>

            <div>
              <label className="label flex items-center">
                <Activity className="mr-1 text-blue-500" size={16} />
                Respiratory Rate (/min)
              </label>
              <input
                type="number"
                name="Resp"
                value={formData.Resp}
                onChange={handleChange}
                className={`input-field ${errors.Resp ? 'border-red-500' : ''}`}
                placeholder="12-20"
                min="5"
                max="60"
              />
              {errors.Resp && <p className="text-red-500 text-sm mt-1">{errors.Resp}</p>}
            </div>

            <div>
              <label className="label">O2 Saturation (%)</label>
              <input
                type="number"
                name="O2Sat"
                value={formData.O2Sat}
                onChange={handleChange}
                className={`input-field ${errors.O2Sat ? 'border-red-500' : ''}`}
                placeholder="95-100"
                min="50"
                max="100"
              />
              {errors.O2Sat && <p className="text-red-500 text-sm mt-1">{errors.O2Sat}</p>}
            </div>
          </div>

          {/* Laboratory Values */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-700 border-b pb-2 flex items-center">
              <Droplets className="mr-2 text-purple-500" size={20} />
              Laboratory Values
            </h3>
            
            <div>
              <label className="label">WBC (K/μL)</label>
              <input
                type="number"
                step="0.1"
                name="WBC"
                value={formData.WBC}
                onChange={handleChange}
                className={`input-field ${errors.WBC ? 'border-red-500' : ''}`}
                placeholder="4.0-11.0"
                min="0.1"
                max="100"
              />
              {errors.WBC && <p className="text-red-500 text-sm mt-1">{errors.WBC}</p>}
            </div>

            <div>
              <label className="label">Hemoglobin (g/dL)</label>
              <input
                type="number"
                step="0.1"
                name="Hgb"
                value={formData.Hgb}
                onChange={handleChange}
                className="input-field"
                placeholder="12-16"
                min="3"
                max="25"
              />
            </div>

            <div>
              <label className="label">Hematocrit (%)</label>
              <input
                type="number"
                name="Hct"
                value={formData.Hct}
                onChange={handleChange}
                className="input-field"
                placeholder="36-48"
                min="10"
                max="70"
              />
            </div>

            <div>
              <label className="label">Platelets (K/μL)</label>
              <input
                type="number"
                name="Platelets"
                value={formData.Platelets}
                onChange={handleChange}
                className="input-field"
                placeholder="150-450"
                min="10"
                max="2000"
              />
            </div>

            <div>
              <label className="label">Lactate (mmol/L)</label>
              <input
                type="number"
                step="0.1"
                name="Lactate"
                value={formData.Lactate}
                onChange={handleChange}
                className={`input-field ${errors.Lactate ? 'border-red-500' : ''}`}
                placeholder="0.5-2.2"
                min="0.1"
                max="30"
              />
              {errors.Lactate && <p className="text-red-500 text-sm mt-1">{errors.Lactate}</p>}
            </div>

            <div>
              <label className="label">Creatinine (mg/dL)</label>
              <input
                type="number"
                step="0.1"
                name="Creatinine"
                value={formData.Creatinine}
                onChange={handleChange}
                className="input-field"
                placeholder="0.6-1.2"
                min="0.1"
                max="20"
              />
            </div>

            <div>
              <label className="label">BUN (mg/dL)</label>
              <input
                type="number"
                name="BUN"
                value={formData.BUN}
                onChange={handleChange}
                className="input-field"
                placeholder="7-20"
                min="1"
                max="200"
              />
            </div>

            <div>
              <label className="label">Glucose (mg/dL)</label>
              <input
                type="number"
                name="Glucose"
                value={formData.Glucose}
                onChange={handleChange}
                className="input-field"
                placeholder="70-100"
                min="20"
                max="1000"
              />
            </div>
          </div>
        </div>

        {/* Additional Lab Values - Collapsible Section */}
        <div className="border-t pt-6">
          <details className="group">
            <summary className="cursor-pointer text-lg font-semibold text-gray-700 mb-4 flex items-center">
              <Brain className="mr-2 text-indigo-500" size={20} />
              Additional Laboratory & Blood Gas Parameters
              <span className="ml-2 text-sm text-gray-500">(Optional - Click to expand)</span>
            </summary>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-4">
              {/* Additional Lab Values */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-600">Additional Labs</h4>
                
                <div>
                  <label className="label text-sm">Total Bilirubin</label>
                  <input
                    type="number"
                    step="0.1"
                    name="Bilirubin_total"
                    value={formData.Bilirubin_total}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="0.2-1.2"
                  />
                </div>

                <div>
                  <label className="label text-sm">AST (U/L)</label>
                  <input
                    type="number"
                    name="AST"
                    value={formData.AST}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="10-40"
                  />
                </div>

                <div>
                  <label className="label text-sm">Alkaline Phos</label>
                  <input
                    type="number"
                    name="Alkalinephos"
                    value={formData.Alkalinephos}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="44-147"
                  />
                </div>
              </div>

              {/* Electrolytes */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-600">Electrolytes</h4>
                
                <div>
                  <label className="label text-sm">Calcium (mg/dL)</label>
                  <input
                    type="number"
                    step="0.1"
                    name="Calcium"
                    value={formData.Calcium}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="8.5-10.5"
                  />
                </div>

                <div>
                  <label className="label text-sm">Chloride (mEq/L)</label>
                  <input
                    type="number"
                    name="Chloride"
                    value={formData.Chloride}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="98-107"
                  />
                </div>

                <div>
                  <label className="label text-sm">Magnesium (mg/dL)</label>
                  <input
                    type="number"
                    step="0.1"
                    name="Magnesium"
                    value={formData.Magnesium}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="1.7-2.2"
                  />
                </div>

                <div>
                  <label className="label text-sm">Phosphate (mg/dL)</label>
                  <input
                    type="number"
                    step="0.1"
                    name="Phosphate"
                    value={formData.Phosphate}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="2.5-4.5"
                  />
                </div>

                <div>
                  <label className="label text-sm">Potassium (mEq/L)</label>
                  <input
                    type="number"
                    step="0.1"
                    name="Potassium"
                    value={formData.Potassium}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="3.5-5.0"
                  />
                </div>
              </div>

              {/* Blood Gas */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-600">Blood Gas</h4>
                
                <div>
                  <label className="label text-sm">pH</label>
                  <input
                    type="number"
                    step="0.01"
                    name="pH"
                    value={formData.pH}
                    onChange={handleChange}
                    className={`input-field text-sm ${errors.pH ? 'border-red-500' : ''}`}
                    placeholder="7.35-7.45"
                  />
                  {errors.pH && <p className="text-red-500 text-xs mt-1">{errors.pH}</p>}
                </div>

                <div>
                  <label className="label text-sm">PaCO2 (mmHg)</label>
                  <input
                    type="number"
                    name="PaCO2"
                    value={formData.PaCO2}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="35-45"
                  />
                </div>

                <div>
                  <label className="label text-sm">Base Excess</label>
                  <input
                    type="number"
                    step="0.1"
                    name="BaseExcess"
                    value={formData.BaseExcess}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="-2 to +2"
                  />
                </div>

                <div>
                  <label className="label text-sm">HCO3 (mEq/L)</label>
                  <input
                    type="number"
                    name="HCO3"
                    value={formData.HCO3}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="22-28"
                  />
                </div>
              </div>

              {/* Respiratory */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-600">Respiratory</h4>
                
                <div>
                  <label className="label text-sm">FiO2 (%)</label>
                  <input
                    type="number"
                    name="FiO2"
                    value={formData.FiO2}
                    onChange={handleChange}
                    className="input-field text-sm"
                    placeholder="21-100"
                  />
                </div>
              </div>
            </div>
          </details>
        </div>

        <div className="flex justify-center pt-6">
          <button
            type="submit"
            disabled={loading}
            className="btn-primary px-8 py-3 text-lg flex items-center"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Analyzing...
              </>
            ) : (
              <>
                <Activity className="mr-2" size={20} />
                Predict Sepsis Risk
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default PatientForm;