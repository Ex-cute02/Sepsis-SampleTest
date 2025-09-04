import React, { useState } from 'react';
import { User, Heart, Thermometer, Activity, Droplets, Stethoscope } from 'lucide-react';

const PatientForm = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    heart_rate: '',
    systolic_bp: '',
    temperature: '',
    respiratory_rate: '',
    wbc_count: '',
    lactate: '',
    sofa_score: ''
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
    
    if (!formData.age || formData.age < 0 || formData.age > 120) {
      newErrors.age = 'Age must be between 0 and 120';
    }
    if (formData.gender === '') {
      newErrors.gender = 'Gender is required';
    }
    if (!formData.heart_rate || formData.heart_rate < 30 || formData.heart_rate > 200) {
      newErrors.heart_rate = 'Heart rate must be between 30 and 200 bpm';
    }
    if (!formData.systolic_bp || formData.systolic_bp < 60 || formData.systolic_bp > 250) {
      newErrors.systolic_bp = 'Systolic BP must be between 60 and 250 mmHg';
    }
    if (!formData.temperature || formData.temperature < 32 || formData.temperature > 45) {
      newErrors.temperature = 'Temperature must be between 32 and 45°C';
    }
    if (!formData.respiratory_rate || formData.respiratory_rate < 5 || formData.respiratory_rate > 60) {
      newErrors.respiratory_rate = 'Respiratory rate must be between 5 and 60 breaths/min';
    }
    if (!formData.wbc_count || formData.wbc_count < 0 || formData.wbc_count > 50) {
      newErrors.wbc_count = 'WBC count must be between 0 and 50 ×10³/μL';
    }
    if (!formData.lactate || formData.lactate < 0 || formData.lactate > 20) {
      newErrors.lactate = 'Lactate must be between 0 and 20 mmol/L';
    }
    if (!formData.sofa_score || formData.sofa_score < 0 || formData.sofa_score > 24) {
      newErrors.sofa_score = 'SOFA score must be between 0 and 24';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      // Convert string values to numbers
      const processedData = {
        ...formData,
        age: parseFloat(formData.age),
        gender: parseInt(formData.gender),
        heart_rate: parseFloat(formData.heart_rate),
        systolic_bp: parseFloat(formData.systolic_bp),
        temperature: parseFloat(formData.temperature),
        respiratory_rate: parseFloat(formData.respiratory_rate),
        wbc_count: parseFloat(formData.wbc_count),
        lactate: parseFloat(formData.lactate),
        sofa_score: parseInt(formData.sofa_score)
      };
      onSubmit(processedData);
    }
  };

  const loadSampleData = () => {
    setFormData({
      age: '65',
      gender: '1',
      heart_rate: '95',
      systolic_bp: '110',
      temperature: '38.2',
      respiratory_rate: '22',
      wbc_count: '12.5',
      lactate: '3.2',
      sofa_score: '4'
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

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Demographics */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-700 border-b pb-2">Demographics</h3>
            
            <div>
              <label className="label">Age (years)</label>
              <input
                type="number"
                name="age"
                value={formData.age}
                onChange={handleChange}
                className={`input-field ${errors.age ? 'border-red-500' : ''}`}
                placeholder="Enter age"
                min="0"
                max="120"
              />
              {errors.age && <p className="text-red-500 text-sm mt-1">{errors.age}</p>}
            </div>

            <div>
              <label className="label">Gender</label>
              <select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                className={`input-field ${errors.gender ? 'border-red-500' : ''}`}
              >
                <option value="">Select gender</option>
                <option value="0">Female</option>
                <option value="1">Male</option>
              </select>
              {errors.gender && <p className="text-red-500 text-sm mt-1">{errors.gender}</p>}
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
                name="heart_rate"
                value={formData.heart_rate}
                onChange={handleChange}
                className={`input-field ${errors.heart_rate ? 'border-red-500' : ''}`}
                placeholder="Enter heart rate"
                min="30"
                max="200"
              />
              {errors.heart_rate && <p className="text-red-500 text-sm mt-1">{errors.heart_rate}</p>}
            </div>

            <div>
              <label className="label">Systolic Blood Pressure (mmHg)</label>
              <input
                type="number"
                name="systolic_bp"
                value={formData.systolic_bp}
                onChange={handleChange}
                className={`input-field ${errors.systolic_bp ? 'border-red-500' : ''}`}
                placeholder="Enter systolic BP"
                min="60"
                max="250"
              />
              {errors.systolic_bp && <p className="text-red-500 text-sm mt-1">{errors.systolic_bp}</p>}
            </div>

            <div>
              <label className="label flex items-center">
                <Thermometer className="mr-1 text-orange-500" size={16} />
                Temperature (°C)
              </label>
              <input
                type="number"
                step="0.1"
                name="temperature"
                value={formData.temperature}
                onChange={handleChange}
                className={`input-field ${errors.temperature ? 'border-red-500' : ''}`}
                placeholder="Enter temperature"
                min="32"
                max="45"
              />
              {errors.temperature && <p className="text-red-500 text-sm mt-1">{errors.temperature}</p>}
            </div>

            <div>
              <label className="label flex items-center">
                <Activity className="mr-1 text-blue-500" size={16} />
                Respiratory Rate (breaths/min)
              </label>
              <input
                type="number"
                name="respiratory_rate"
                value={formData.respiratory_rate}
                onChange={handleChange}
                className={`input-field ${errors.respiratory_rate ? 'border-red-500' : ''}`}
                placeholder="Enter respiratory rate"
                min="5"
                max="60"
              />
              {errors.respiratory_rate && <p className="text-red-500 text-sm mt-1">{errors.respiratory_rate}</p>}
            </div>
          </div>

          {/* Laboratory Values */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-700 border-b pb-2 flex items-center">
              <Droplets className="mr-2 text-purple-500" size={20} />
              Laboratory Values
            </h3>
            
            <div>
              <label className="label">White Blood Cell Count (×10³/μL)</label>
              <input
                type="number"
                step="0.1"
                name="wbc_count"
                value={formData.wbc_count}
                onChange={handleChange}
                className={`input-field ${errors.wbc_count ? 'border-red-500' : ''}`}
                placeholder="Enter WBC count"
                min="0"
                max="50"
              />
              {errors.wbc_count && <p className="text-red-500 text-sm mt-1">{errors.wbc_count}</p>}
            </div>

            <div>
              <label className="label">Lactate (mmol/L)</label>
              <input
                type="number"
                step="0.1"
                name="lactate"
                value={formData.lactate}
                onChange={handleChange}
                className={`input-field ${errors.lactate ? 'border-red-500' : ''}`}
                placeholder="Enter lactate level"
                min="0"
                max="20"
              />
              {errors.lactate && <p className="text-red-500 text-sm mt-1">{errors.lactate}</p>}
            </div>
          </div>

          {/* Clinical Scores */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-700 border-b pb-2 flex items-center">
              <Stethoscope className="mr-2 text-green-500" size={20} />
              Clinical Assessment
            </h3>
            
            <div>
              <label className="label">SOFA Score (0-24)</label>
              <input
                type="number"
                name="sofa_score"
                value={formData.sofa_score}
                onChange={handleChange}
                className={`input-field ${errors.sofa_score ? 'border-red-500' : ''}`}
                placeholder="Enter SOFA score"
                min="0"
                max="24"
              />
              {errors.sofa_score && <p className="text-red-500 text-sm mt-1">{errors.sofa_score}</p>}
              <p className="text-sm text-gray-500 mt-1">
                Sequential Organ Failure Assessment score
              </p>
            </div>
          </div>
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