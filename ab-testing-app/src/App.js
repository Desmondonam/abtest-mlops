import React from 'react';
import './App.css';
import ABTestInterface from './components/ABTestInterface'; // Import the new component

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>A/B Testing Application</h1>
        <p>Welcome to your A/B Testing Interface!</p>
      </header>
      <main style={{ padding: '20px' }}>
        <ABTestInterface /> {/* Use the new component here */}
      </main>
    </div>
  );
}

export default App;