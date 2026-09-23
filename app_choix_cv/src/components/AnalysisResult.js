export default function AnalysisResult({ result }) {
  const { prediction, skills, experience_years, quality_score, word_count } = result;

  return (
    <div className="result">
      <div className="result-header">
        <div>
          <p className="result-category-label">Catégorie prédite</p>
          <p className="result-category">{prediction.predicted_category}</p>
        </div>
        <div className="result-score">
          <div className="score-value">{quality_score}</div>
          <div className="score-label">score sur 100</div>
        </div>
      </div>

      <div className="predictions-list">
        {prediction.top_predictions.map((pred) => (
          <div className="prediction-row" key={pred.category}>
            <span className="pred-name">{pred.category}</span>
            <span className="pred-pct">{pred.probability.toFixed(1)}%</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${pred.probability}%` }} />
            </div>
          </div>
        ))}
      </div>

      <div className="detail-grid">
        <div className="detail-card">
          <p className="detail-label">Expérience estimée</p>
          <p className="detail-value">{experience_years} an{experience_years > 1 ? 's' : ''}</p>
        </div>
        <div className="detail-card">
          <p className="detail-label">Compétences repérées</p>
          <p className="detail-value">{skills.length}</p>
        </div>
        <div className="detail-card">
          <p className="detail-label">Mots</p>
          <p className="detail-value">{word_count}</p>
        </div>
      </div>

      <div>
        <p className="skills-title">Compétences identifiées</p>
        {skills.length > 0 ? (
          <div className="skills-chips">
            {skills.map((skill) => (
              <span className="chip" key={skill}>{skill}</span>
            ))}
          </div>
        ) : (
          <p className="no-skills">Aucun mot-clé de compétence reconnu dans le texte.</p>
        )}
      </div>
    </div>
  );
}
