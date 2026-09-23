import { useState } from 'react';
import FileDropzone from './FileDropzone';
import { compareFiles, compareTexts } from '../api';

const PASTE_PLACEHOLDER = "Colle le texte d'un CV par bloc, séparé par une ligne \"---\"";

function splitPastedTexts(raw) {
  return raw
    .split(/\n\s*---\s*\n/)
    .map((chunk) => chunk.trim())
    .filter(Boolean);
}

export default function ComparePanel() {
  const [mode, setMode] = useState('file');
  const [files, setFiles] = useState([]);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const pastedCount = mode === 'text' ? splitPastedTexts(text).length : 0;
  const canSubmit = mode === 'file' ? files.length >= 2 : pastedCount >= 2;

  function switchMode(nextMode) {
    setMode(nextMode);
    setError(null);
  }

  async function handleSubmit() {
    if (!canSubmit || loading) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const result = mode === 'file' ? await compareFiles(files) : await compareTexts(splitPastedTexts(text));
      setData(result);
    } catch (err) {
      setError(err.message || 'Erreur lors de la comparaison');
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFiles([]);
    setText('');
    setData(null);
    setError(null);
  }

  return (
    <div className="panel">
      <h2>Comparer plusieurs CV</h2>
      <p className="panel-hint">
        Ajoute au moins deux CV : ils sont classés par score de qualité, du meilleur au moins bon.
      </p>

      <div className="mode-switch">
        <button
          type="button"
          className={mode === 'file' ? 'active' : ''}
          onClick={() => switchMode('file')}
        >
          Fichiers
        </button>
        <button
          type="button"
          className={mode === 'text' ? 'active' : ''}
          onClick={() => switchMode('text')}
        >
          Texte
        </button>
      </div>

      {mode === 'file' ? (
        <FileDropzone multiple files={files} onFilesChange={setFiles} />
      ) : (
        <textarea
          className="cv-textarea"
          placeholder={PASTE_PLACEHOLDER}
          value={text}
          onChange={(e) => setText(e.target.value)}
          style={{ minHeight: 220 }}
        />
      )}

      {mode === 'text' && (
        <p className="dz-hint" style={{ marginTop: 8 }}>
          {pastedCount} CV détecté{pastedCount > 1 ? 's' : ''}
        </p>
      )}

      <button className="btn-primary" onClick={handleSubmit} disabled={!canSubmit || loading}>
        {loading ? 'Comparaison en cours…' : 'Comparer les CV'}
      </button>

      {loading && (
        <div className="loading-line">
          <span className="loading-dot" />
          Analyse de chaque CV et classement en cours…
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {data && (
        <div style={{ marginTop: 28 }}>
          {data.results.length === 0 ? (
            <p className="empty-state">Aucun CV n'a pu être analysé.</p>
          ) : (
            <table className="ledger">
              <thead>
                <tr>
                  <th>Rang</th>
                  <th>CV</th>
                  <th>Score</th>
                  <th>Confiance</th>
                  <th>Compétences</th>
                  <th>Expérience</th>
                </tr>
              </thead>
              <tbody>
                {data.results.map((row) => (
                  <tr key={row.cv_id}>
                    <td className="rank-cell">{String(row.rank).padStart(2, '0')}</td>
                    <td>
                      <div className="cv-name">{row.filename || `CV ${row.cv_id}`}</div>
                      <div className="cv-category">{row.category}</div>
                    </td>
                    <td className="num-cell score-cell">{row.quality_score}</td>
                    <td className="num-cell">{row.confidence.toFixed(1)}%</td>
                    <td className="num-cell">{row.skills_count}</td>
                    <td className="num-cell">
                      {row.experience_years} an{row.experience_years > 1 ? 's' : ''}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {data.errors && data.errors.length > 0 && (
            <details className="compare-errors">
              <summary>
                {data.errors.length} fichier{data.errors.length > 1 ? 's' : ''} n'ont pas pu être lu{data.errors.length > 1 ? 's' : ''}
              </summary>
              <ul>
                {data.errors.map((e) => (
                  <li key={e.filename}>{e.filename} — {e.error}</li>
                ))}
              </ul>
            </details>
          )}

          <button className="btn-text" onClick={reset} style={{ marginTop: 16 }}>
            Nouvelle comparaison
          </button>
        </div>
      )}
    </div>
  );
}
