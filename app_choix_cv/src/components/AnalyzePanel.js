import { useState } from 'react';
import FileDropzone from './FileDropzone';
import AnalysisResult from './AnalysisResult';
import { analyzeFile, analyzeText } from '../api';

export default function AnalyzePanel() {
  const [mode, setMode] = useState('file'); // 'file' | 'text'
  const [files, setFiles] = useState([]);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const canSubmit = mode === 'file' ? files.length === 1 : text.trim().length > 0;

  function switchMode(nextMode) {
    setMode(nextMode);
    setError(null);
  }

  async function handleSubmit() {
    if (!canSubmit || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = mode === 'file' ? await analyzeFile(files[0]) : await analyzeText(text);
      setResult(data);
    } catch (err) {
      setError(err.message || "Erreur lors de l'analyse");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFiles([]);
    setText('');
    setResult(null);
    setError(null);
  }

  return (
    <div className="panel">
      <h2>Analyser un CV</h2>
      <p className="panel-hint">
        Dépose un fichier PDF ou DOCX, ou colle directement le texte du CV.
      </p>

      <div className="mode-switch">
        <button
          type="button"
          className={mode === 'file' ? 'active' : ''}
          onClick={() => switchMode('file')}
        >
          Fichier
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
        <FileDropzone multiple={false} files={files} onFilesChange={setFiles} />
      ) : (
        <textarea
          className="cv-textarea"
          placeholder="Colle ici le contenu du CV…"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      )}

      <button className="btn-primary" onClick={handleSubmit} disabled={!canSubmit || loading}>
        {loading ? 'Analyse en cours…' : 'Analyser le CV'}
      </button>

      {loading && (
        <div className="loading-line">
          <span className="loading-dot" />
          Lecture du fichier et classification en cours…
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div style={{ marginTop: 28 }}>
          <AnalysisResult result={result} />
          <button className="btn-text" onClick={reset} style={{ marginTop: 12 }}>
            Analyser un autre CV
          </button>
        </div>
      )}
    </div>
  );
}
