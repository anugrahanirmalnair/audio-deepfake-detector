import { useCallback, useEffect, useState } from "react";
import { analyzeAudio, checkHealth } from "./api";
import { ResultsPanel } from "./components/ResultsPanel";
import { WaveformHeatmap } from "./components/WaveformHeatmap";
import type { AnalyzeResponse } from "./types";

const ACCEPT = ".wav,.flac,.mp3,.m4a,.ogg,.webm,audio/*";

export default function App() {
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkHealth().then(setApiOk);
  }, []);

  useEffect(() => {
    if (!file) {
      setAudioUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const picked = e.target.files?.[0];
      if (!picked) return;
      setFile(picked);
      setResult(null);
      setError(null);
    },
    [],
  );

  const onAnalyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeAudio(file);
      setResult(data);
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }, [file]);

  return (
    <div className="app">
      <header className="header">
        <div>
          <p className="eyebrow">Neural codec analysis</p>
          <h1>Codec Deepfake Detector</h1>
          <p className="subtitle">
            DAC reconstruction + token features · Random Forest classifier
          </p>
        </div>
        <div className={`api-badge ${apiOk ? "ok" : apiOk === false ? "down" : ""}`}>
          API {apiOk === null ? "…" : apiOk ? "connected" : "offline"}
        </div>
      </header>

      <section className="upload-section">
        <label className="upload-zone">
          <input
            type="file"
            accept={ACCEPT}
            onChange={onFileChange}
            disabled={loading}
          />
          <span className="upload-icon">♪</span>
          <span className="upload-title">
            {file ? file.name : "Drop or choose an audio file"}
          </span>
          <span className="upload-hint">WAV, FLAC, MP3, M4A, OGG, WebM</span>
        </label>

        <button
          type="button"
          className="analyze-btn"
          onClick={onAnalyze}
          disabled={!file || loading || apiOk === false}
        >
          {loading ? "Analyzing…" : "Analyze audio"}
        </button>

        {error && <p className="error-banner">{error}</p>}
      </section>

      {result && (
        <>
          <ResultsPanel result={result} />
          <WaveformHeatmap audioUrl={audioUrl} windows={result.windows} />
        </>
      )}

      {!result && !loading && (
        <p className="placeholder">
          Upload a clip and run analysis to see verdict, scores, and the
          per-window reconstruction heatmap.
        </p>
      )}
    </div>
  );
}
