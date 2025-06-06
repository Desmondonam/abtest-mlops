import React, { useState } from 'react';

function ABTestInterface() {
  // State to manage which model is selected (e.g., 'logistic', 'decision_tree')
  const [selectedModel, setSelectedModel] = useState('');
  // State to store the A/B test results
  const [abTestResults, setAbTestResults] = useState(null);
  // State for loading indicator
  const [isLoading, setIsLoading] = useState(false);
  // State for error messages
  const [error, setError] = useState(null);

  const handleModelSelection = (model) => {
    setSelectedModel(model);
    setAbTestResults(null); // Clear previous results when model changes
    setError(null); // Clear previous errors
  };

  const runABTest = async () => {
    if (!selectedModel) {
      setError("Please select a model first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setAbTestResults(null);

    try {
      // In a real application, you'd send a request to your Python backend here.
      // For now, let's simulate a network request.
      console.log(`Running A/B test with model: ${selectedModel}`);

      // Simulate API call
      const response = await new Promise(resolve => setTimeout(() => {
        if (Math.random() > 0.1) { // Simulate success 90% of the time
            resolve({
                success: true,
                model: selectedModel,
                results: {
                    accuracy: 0.85,
                    confusion_matrix: [[100, 10], [5, 85]],
                    message: `A/B test completed successfully using the ${selectedModel} model.`,
                    // In a real app, you'd get the actual best model ID from MLflow
                    best_model_id: "your_mlflow_run_id_here"
                }
            });
        } else { // Simulate failure 10% of the time
            resolve({
                success: false,
                message: "Failed to run A/B test. Please try again."
            });
        }
      }, 2000)); // Simulate a 2-second delay

      if (response.success) {
        setAbTestResults(response.results);
      } else {
        setError(response.message);
      }

    } catch (err) {
      setError("An unexpected error occurred. Please check your network connection.");
      console.error("API call error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <h2>A/B Testing with MLflow Models</h2>

      <div>
        <h3>Select a Model:</h3>
        <button
          onClick={() => handleModelSelection('logistic_regression')}
          style={{ backgroundColor: selectedModel === 'logistic_regression' ? '#007bff' : '#f0f0f0', color: selectedModel === 'logistic_regression' ? 'white' : 'black', padding: '10px', margin: '5px', borderRadius: '5px', border: 'none', cursor: 'pointer' }}
        >
          Logistic Regression
        </button>
        <button
          onClick={() => handleModelSelection('decision_tree')}
          style={{ backgroundColor: selectedModel === 'decision_tree' ? '#007bff' : '#f0f0f0', color: selectedModel === 'decision_tree' ? 'white' : 'black', padding: '10px', margin: '5px', borderRadius: '5px', border: 'none', cursor: 'pointer' }}
        >
          Decision Tree
        </button>
      </div>

      {selectedModel && (
        <p>Selected Model: <strong>{selectedModel.replace('_', ' ').toUpperCase()}</strong></p>
      )}

      <button
        onClick={runABTest}
        disabled={isLoading || !selectedModel}
        style={{ padding: '10px 20px', fontSize: '16px', marginTop: '20px', cursor: isLoading ? 'not-allowed' : 'pointer' }}
      >
        {isLoading ? 'Running A/B Test...' : 'Run A/B Test'}
      </button>

      {error && <p style={{ color: 'red', marginTop: '10px' }}>Error: {error}</p>}

      {abTestResults && (
        <div style={{ marginTop: '30px', borderTop: '1px solid #ccc', paddingTop: '20px' }}>
          <h3>A/B Test Results ({abTestResults.model.replace('_', ' ').toUpperCase()})</h3>
          <p>Message: {abTestResults.message}</p>
          <p>Accuracy: <strong>{(abTestResults.accuracy * 100).toFixed(2)}%</strong></p>
          <p>Confusion Matrix:</p>
          <pre>{JSON.stringify(abTestResults.confusion_matrix, null, 2)}</pre>
          <p>MLflow Best Model ID: <code>{abTestResults.best_model_id}</code></p>
          {/* You would render more detailed results here, like charts */}
        </div>
      )}
    </div>
  );
}

export default ABTestInterface;