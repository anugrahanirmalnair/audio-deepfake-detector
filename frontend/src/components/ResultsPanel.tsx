import type { AnalyzeResponse } from "../types";

interface Props {
  result: AnalyzeResponse;
}

export function ResultsPanel({ result }: Props) {
  const isFake = result.verdict === "FAKE";

  return (
    <div className={`results-panel ${isFake ? "fake" : "real"}`}>
      <div className="verdict-block">
        <span className="verdict-label">Verdict</span>
        <span className={`verdict-value ${isFake ? "fake" : "real"}`}>
          {result.verdict}
        </span>
      </div>

      <div className="metrics-grid">
        <Metric
          label="Authenticity score"
          value={result.authenticity_score}
          format="percent"
        />
        <Metric
          label="Spoof probability"
          value={result.spoof_probability}
          format="percent"
        />
        <Metric
          label="Confidence"
          value={result.confidence}
          format="percent"
        />
        <Metric
          label="Windows analyzed"
          value={result.windows.length}
          format="int"
        />
      </div>

      <div className="bar-group">
        <label>Authenticity</label>
        <div className="bar-track">
          <div
            className="bar-fill auth"
            style={{ width: `${result.authenticity_score * 100}%` }}
          />
        </div>
      </div>
      <div className="bar-group">
        <label>Spoof likelihood</label>
        <div className="bar-track">
          <div
            className="bar-fill spoof"
            style={{ width: `${result.spoof_probability * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  format,
}: {
  label: string;
  value: number;
  format: "percent" | "int";
}) {
  const display =
    format === "percent" ? `${(value * 100).toFixed(1)}%` : String(value);

  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{display}</span>
    </div>
  );
}
