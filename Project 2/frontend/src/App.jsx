import React, { useState } from 'react';
import { 
  Database, 
  BarChart2, 
  Cpu, 
  Activity, 
  Award,
  CheckCircle2,
  ArrowRight
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

const API_BASE = 'http://localhost:8000/api';

const steps = [
  { id: 1, title: 'Load & Preprocess', icon: Database, endpoint: '/step1_load' },
  { id: 2, title: 'Feature Engineering', icon: BarChart2, endpoint: '/step2_features' },
  { id: 3, title: 'Train Models', icon: Cpu, endpoint: '/step3_train' },
  { id: 4, title: 'Evaluate & Compare', icon: Activity, endpoint: '/step4_evaluate' },
  { id: 5, title: 'Best Model Analysis', icon: Award, endpoint: '/step5_best_model' }
];

function App() {
  const [currentStep, setCurrentStep] = useState(1);
  const [completedSteps, setCompletedSteps] = useState([]);
  const [stepData, setStepData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchStepData = async (stepId) => {
    const step = steps.find(s => s.id === stepId);
    if (!step) return;

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}${step.endpoint}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch data for step ${stepId}`);
      }
      const data = await response.json();
      setStepData(prev => ({ ...prev, [stepId]: data }));
      if (!completedSteps.includes(stepId)) {
        setCompletedSteps(prev => [...prev, stepId]);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleNextStep = () => {
    if (currentStep < steps.length) {
      const nextStep = currentStep + 1;
      setCurrentStep(nextStep);
      if (!stepData[nextStep]) {
        fetchStepData(nextStep);
      }
    }
  };

  // Step renderers
  const renderStep1 = (data) => (
    <div>
      <div className="explanation-box">
        <p>{data.preprocessing_explanation}</p>
      </div>
      <div className="data-grid">
        <div className="stat-box">
          <div className="stat-label">Total Rows</div>
          <div className="stat-value">{data.dataset_shape[0]}</div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Total Features</div>
          <div className="stat-value">{data.dataset_shape[1]}</div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Training Samples</div>
          <div className="stat-value">{data.train_size}</div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Testing Samples</div>
          <div className="stat-value">{data.test_size}</div>
        </div>
      </div>
      
      <div className="data-card" style={{marginTop: '2rem'}}>
        <h3>Features Available</h3>
        <div style={{display: 'flex', flexWrap: 'wrap', gap: '0.5rem'}}>
          {data.features.map(f => (
            <span key={f} style={{background: '#e2e8f0', padding: '0.25rem 0.75rem', borderRadius: '1rem', fontSize: '0.875rem'}}>
              {f}
            </span>
          ))}
        </div>
      </div>
    </div>
  );

  const renderStep2 = (data) => {
    // Sort features by importance for chart
    const sortedFeatures = [...data.feature_analysis].sort((a, b) => b.importance - a.importance);
    
    return (
      <div>
        <div className="explanation-box">
          <p>{data.explanation}</p>
        </div>
        
        <div className="data-card">
          <h3>Dropped Features (Least Important)</h3>
          <div style={{display: 'flex', gap: '1rem', marginBottom: '1rem'}}>
            {data.dropped_features.map(f => (
              <div key={f} style={{background: '#fee2e2', color: '#991b1b', padding: '0.5rem 1rem', borderRadius: '8px', fontWeight: '500'}}>
                - {f}
              </div>
            ))}
          </div>
        </div>

        <div className="data-card" style={{height: '400px'}}>
          <h3>Feature Importance Ranking</h3>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sortedFeatures} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="feature" type="category" width={100} />
              <RechartsTooltip />
              <Bar dataKey="importance" fill="#2563eb" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  const renderStep3 = (data) => (
    <div>
      <div className="explanation-box">
        <p>{data.explanation}</p>
      </div>
      
      <div className="data-grid" style={{marginTop: '2rem'}}>
        {data.trained_models.map((model, index) => (
          <div key={model} className="stat-box" style={{textAlign: 'center', padding: '2rem 1rem'}}>
            <Cpu size={48} color="#2563eb" style={{marginBottom: '1rem'}} />
            <h3 style={{fontSize: '1.25rem', marginBottom: '0.5rem'}}>{model}</h3>
            <span style={{color: '#10b981', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', fontWeight: '600'}}>
              <CheckCircle2 size={18} /> Trained Successfully
            </span>
          </div>
        ))}
      </div>
    </div>
  );

  const renderStep4 = (data) => {
    const models = Object.keys(data.metrics);
    // Convert to array for Recharts
    const chartData = models.map(m => ({
      name: m,
      Accuracy: data.metrics[m].accuracy * 100,
      F1_Score: data.metrics[m].f1_score * 100,
    }));

    // Find best overall (for highlighting)
    let bestModel = models[0];
    let maxF1 = data.metrics[bestModel].f1_score;
    models.forEach(m => {
      if(data.metrics[m].f1_score > maxF1) {
        maxF1 = data.metrics[m].f1_score;
        bestModel = m;
      }
    });

    return (
      <div>
        <div className="explanation-box">
          <p>{data.explanation}</p>
        </div>

        <table className="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
            </tr>
          </thead>
          <tbody>
            {models.map(m => (
              <tr key={m} className={m === bestModel ? 'highlight-row' : ''}>
                <td>{m} {m === bestModel && '👑'}</td>
                <td>{(data.metrics[m].accuracy * 100).toFixed(1)}%</td>
                <td>{(data.metrics[m].precision * 100).toFixed(1)}%</td>
                <td>{(data.metrics[m].recall * 100).toFixed(1)}%</td>
                <td>{(data.metrics[m].f1_score * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="data-card" style={{height: '350px', marginTop: '2rem'}}>
          <h3>Performance Comparison (Accuracy & F1)</h3>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <RechartsTooltip formatter={(value) => value.toFixed(1) + '%'} />
              <Legend />
              <Bar dataKey="Accuracy" fill="#2563eb" radius={[4, 4, 0, 0]} />
              <Bar dataKey="F1_Score" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  const renderStep5 = (data) => (
    <div>
      <div className="data-card" style={{textAlign: 'center', backgroundColor: '#eff6ff', borderColor: '#bfdbfe'}}>
        <h2 style={{color: '#1d4ed8', fontSize: '2rem', marginBottom: '0.5rem'}}>Best Model: {data.best_model}</h2>
        <p style={{color: '#3b82f6', fontSize: '1.125rem'}}>Achieved {(data.metrics.f1_score * 100).toFixed(1)}% F1-Score</p>
      </div>

      <div className="explanation-box">
        <p>{data.conclusion}</p>
      </div>

      <div className="data-card">
        <h3 style={{textAlign: 'center'}}>Confusion Matrix (Test Data)</h3>
        <p style={{textAlign: 'center', color: '#64748b', fontSize: '0.875rem', marginBottom: '1rem'}}>
          How the model predicted versus reality
        </p>
        
        <div className="confusion-matrix">
          <div className="cm-cell cm-header" style={{background: 'transparent'}}></div>
          <div className="cm-cell cm-header">Predicted: No Disease</div>
          <div className="cm-cell cm-header">Predicted: Disease</div>
          
          <div className="cm-cell cm-header">Actual: No Disease</div>
          <div className="cm-cell">
            <span className="cm-value">{data.confusion_matrix[0][0]}</span>
            <span className="cm-label">True Negatives</span>
          </div>
          <div className="cm-cell" style={{backgroundColor: '#fef2f2'}}>
            <span className="cm-value" style={{color: '#dc2626'}}>{data.confusion_matrix[0][1]}</span>
            <span className="cm-label">False Positives (Error)</span>
          </div>
          
          <div className="cm-cell cm-header">Actual: Disease</div>
          <div className="cm-cell" style={{backgroundColor: '#fef2f2'}}>
            <span className="cm-value" style={{color: '#dc2626'}}>{data.confusion_matrix[1][0]}</span>
            <span className="cm-label">False Negatives (Error)</span>
          </div>
          <div className="cm-cell" style={{backgroundColor: '#f0fdf4'}}>
            <span className="cm-value" style={{color: '#166534'}}>{data.confusion_matrix[1][1]}</span>
            <span className="cm-label">True Positives</span>
          </div>
        </div>
      </div>
    </div>
  );

  const currentStepInfo = steps.find(s => s.id === currentStep);

  return (
    <div className="app-container">
      <header className="header">
        <h1>Heart Disease ML Pipeline</h1>
        <p>A professional end-to-end machine learning project analysis</p>
      </header>

      <div className="layout">
        {/* Sidebar Stepper */}
        <aside className="stepper">
          {steps.map(step => {
            const Icon = step.icon;
            const isActive = currentStep === step.id;
            const isCompleted = completedSteps.includes(step.id) && !isActive;
            
            return (
              <div 
                key={step.id}
                className={`step-item ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
                onClick={() => {
                  if (completedSteps.includes(step.id) || step.id === currentStep) {
                    setCurrentStep(step.id);
                  }
                }}
              >
                <div className="step-number">
                  {isCompleted ? <CheckCircle2 size={16} /> : step.id}
                </div>
                <div style={{flex: 1}}>{step.title}</div>
                <Icon size={18} />
              </div>
            );
          })}
        </aside>

        {/* Main Content */}
        <main className="content-area">
          <h2 className="step-title">Step {currentStep}: {currentStepInfo?.title}</h2>
          
          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}

          {loading ? (
            <div className="loading">
              <div className="spinner"></div>
              <p>Processing ML Pipeline...</p>
            </div>
          ) : (
            <div>
              {!stepData[currentStep] && currentStep === 1 && (
                <div style={{textAlign: 'center', padding: '3rem 0'}}>
                  <Database size={64} color="#94a3b8" style={{marginBottom: '1rem'}} />
                  <h3 style={{marginBottom: '1rem'}}>Ready to start the ML pipeline?</h3>
                  <button className="btn" onClick={() => fetchStepData(1)}>
                    Load Dataset & Preprocess
                  </button>
                </div>
              )}

              {stepData[currentStep] && (
                <>
                  {currentStep === 1 && renderStep1(stepData[1])}
                  {currentStep === 2 && renderStep2(stepData[2])}
                  {currentStep === 3 && renderStep3(stepData[3])}
                  {currentStep === 4 && renderStep4(stepData[4])}
                  {currentStep === 5 && renderStep5(stepData[5])}
                  
                  {currentStep < steps.length && (
                    <div style={{marginTop: '3rem', display: 'flex', justifyContent: 'flex-end', borderTop: '1px solid #e2e8f0', paddingTop: '1.5rem'}}>
                      <button className="btn" onClick={handleNextStep}>
                        Proceed to Next Step <ArrowRight size={18} style={{marginLeft: '0.5rem'}} />
                      </button>
                    </div>
                  )}
                  {currentStep === steps.length && (
                    <div style={{marginTop: '3rem', textAlign: 'center', padding: '2rem', background: '#f8fafc', borderRadius: '8px'}}>
                      <Award size={48} color="#10b981" style={{marginBottom: '1rem'}} />
                      <h3 style={{fontSize: '1.5rem', color: '#0f172a'}}>Analysis Complete!</h3>
                      <p style={{color: '#64748b'}}>You have successfully run the entire machine learning pipeline.</p>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
