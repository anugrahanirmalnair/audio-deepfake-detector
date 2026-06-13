/** Matches backend POST /analyze response exactly. */

export interface WindowScore {
  start_ms: number;
  end_ms: number;
  recon_score: number;
  token_score: number;
}

export interface AnalyzeResponse {
  authenticity_score: number;
  spoof_probability: number;
  verdict: "REAL" | "FAKE";
  confidence: number;
  windows: WindowScore[];
}
