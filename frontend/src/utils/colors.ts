/** Map recon_score to green (low) → red (high) within clip range. */
export function reconHeatColor(
  score: number,
  min: number,
  max: number,
): string {
  const span = max - min || 1;
  const t = Math.min(1, Math.max(0, (score - min) / span));
  const r = Math.round(34 + t * (220 - 34));
  const g = Math.round(180 - t * (180 - 50));
  const b = Math.round(90 - t * (90 - 50));
  const alpha = 0.35 + t * 0.45;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function reconBorderColor(
  score: number,
  min: number,
  max: number,
): string {
  const span = max - min || 1;
  const t = Math.min(1, Math.max(0, (score - min) / span));
  const r = Math.round(34 + t * (220 - 34));
  const g = Math.round(180 - t * (180 - 50));
  const b = Math.round(90 - t * (90 - 50));
  return `rgb(${r}, ${g}, ${b})`;
}
