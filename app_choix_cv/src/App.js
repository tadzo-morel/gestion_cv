import { useEffect, useState } from 'react';
import './App.css';
import AnalyzePanel from './components/AnalyzePanel';
import ComparePanel from './components/ComparePanel';
import { fetchStats } from './api';

function App() {
  const [tab, setTab] = useState('analyze');
  const [stats, setStats] = useState(null);
  const [statsError, setStatsError] = useState(null);

  useEffect(() => {
    fetchStats()
      .then(setStats)
      .catch((err) => setStatsError(err.message));
  }, []);

  return (
    <div className="page">
      <header className="masthead">
        <h1>Dossier CV</h1>
        <p className="subtitle">
          Classement automatique de CV par catégorie, à partir d'un modèle entraîné sur un
          jeu de données de CV réels.
        </p>

        {stats && (
          <div className="stats-strip">
            <span><strong>{stats.total_cvs}</strong> CV d'entraînement</span>
            <span><strong>{stats.categories}</strong> catégories</span>
            <span><strong>{stats.model_accuracy}%</strong> de précision ({stats.model_name})</span>
          </div>
        )}
        {statsError && (
          <div className="stats-strip">
            Serveur d'analyse injoignable — vérifie que <code>python app.py</code> tourne sur
            le port 5000.
          </div>
        )}
      </header>

      <nav className="tabs">
        <button
          type="button"
          className={`tab-btn${tab === 'analyze' ? ' active' : ''}`}
          onClick={() => setTab('analyze')}
        >
          Analyser
        </button>
        <button
          type="button"
          className={`tab-btn${tab === 'compare' ? ' active' : ''}`}
          onClick={() => setTab('compare')}
        >
          Comparer
        </button>
      </nav>

      {tab === 'analyze' ? <AnalyzePanel /> : <ComparePanel />}

      <footer className="page-footer">
        <span>TPE — Morel Tadzo, Licence 3 Informatique</span>
        <span>Modèle Random Forest sur vectorisation TF-IDF</span>
      </footer>
    </div>
  );
}

export default App;
