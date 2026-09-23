import { useRef, useState } from 'react';

const ACCEPTED_EXTENSIONS = ['pdf', 'docx', 'txt'];

function getExtension(filename) {
  const parts = filename.split('.');
  return parts.length > 1 ? parts.pop().toLowerCase() : '';
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} o`;
  return `${(bytes / 1024).toFixed(1)} Ko`;
}

/**
 * Zone de dépôt de fichier(s). En mode `multiple`, accumule les fichiers
 * ajoutés successivement plutôt que de remplacer la sélection.
 */
export default function FileDropzone({ multiple = false, files, onFilesChange }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  function addFiles(fileList) {
    const incoming = Array.from(fileList).filter((f) =>
      ACCEPTED_EXTENSIONS.includes(getExtension(f.name))
    );
    if (incoming.length === 0) return;

    if (multiple) {
      onFilesChange([...files, ...incoming]);
    } else {
      onFilesChange([incoming[0]]);
    }
  }

  function handleInputChange(e) {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(e.target.files);
    }
    e.target.value = '';
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  }

  function removeFile(index) {
    onFilesChange(files.filter((_, i) => i !== index));
  }

  const showDropzone = multiple || files.length === 0;

  return (
    <div>
      {files.length > 0 && (
        <div className="file-list" style={{ marginBottom: showDropzone ? 12 : 0 }}>
          {files.map((file, index) => (
            <div className="file-chip" key={`${file.name}-${index}`}>
              <div className="file-name">{file.name}</div>
              <div className="file-meta">{formatSize(file.size)}</div>
              <button type="button" onClick={() => removeFile(index)} aria-label={`Retirer ${file.name}`}>
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {showDropzone && (
        <div
          className={`dropzone${dragOver ? ' dragover' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current && inputRef.current.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            multiple={multiple}
            onChange={handleInputChange}
          />
          <p className="dz-title">
            {multiple && files.length > 0
              ? 'Ajouter un autre CV'
              : 'Glissez un CV ici, ou cliquez pour parcourir'}
          </p>
          <p className="dz-hint">
            {multiple ? 'Vous pouvez ajouter plusieurs fichiers' : 'Un seul fichier à la fois'}
          </p>
          <p className="dz-formats">PDF · DOCX · TXT</p>
        </div>
      )}
    </div>
  );
}
