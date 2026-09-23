// Client API pour le backend Flask.
//
// En développement, package.json déclare "proxy": "http://localhost:5000",
// donc les chemins relatifs /api/... sont automatiquement transmis au
// serveur Flask par le serveur de dev de Create React App (pas de souci
// de CORS à gérer en local).

async function parseJsonSafe(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function handleResponse(response) {
  const data = await parseJsonSafe(response);
  if (!response.ok) {
    const message = data && data.error ? data.error : `Erreur serveur (${response.status})`;
    throw new Error(message);
  }
  return data;
}

export async function fetchStats() {
  const response = await fetch('/api/stats');
  return handleResponse(response);
}

export async function analyzeText(cvText) {
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cv_text: cvText }),
  });
  return handleResponse(response);
}

export async function analyzeFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch('/api/analyze-file', {
    method: 'POST',
    body: formData,
  });
  return handleResponse(response);
}

export async function compareTexts(cvTexts) {
  const response = await fetch('/api/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cvs: cvTexts }),
  });
  return handleResponse(response);
}

export async function compareFiles(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  const response = await fetch('/api/compare-files', {
    method: 'POST',
    body: formData,
  });
  return handleResponse(response);
}
