import { useEffect, useRef, useState } from "react";
import type { WindowScore } from "../types";
import { reconBorderColor, reconHeatColor } from "../utils/colors";

interface Props {
  audioUrl: string | null;
  windows: WindowScore[];
}

function drawWaveform(
  canvas: HTMLCanvasElement,
  audioBuffer: AudioBuffer,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, width, height);

  const data = audioBuffer.getChannelData(0);
  const step = Math.ceil(data.length / width);
  const mid = height / 2;
  const amp = height * 0.38;

  ctx.fillStyle = "#1a2332";
  ctx.fillRect(0, 0, width, height);

  ctx.beginPath();
  ctx.moveTo(0, mid);

  for (let x = 0; x < width; x++) {
    let min = 1;
    let max = -1;
    const start = x * step;
    for (let i = 0; i < step && start + i < data.length; i++) {
      const v = data[start + i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.lineTo(x, mid + min * amp);
  }

  for (let x = width - 1; x >= 0; x--) {
    let max = -1;
    const start = x * step;
    for (let i = 0; i < step && start + i < data.length; i++) {
      const v = data[start + i];
      if (v > max) max = v;
    }
    ctx.lineTo(x, mid + max * amp);
  }

  ctx.closePath();
  ctx.fillStyle = "rgba(148, 163, 184, 0.55)";
  ctx.fill();

  ctx.strokeStyle = "rgba(226, 232, 240, 0.85)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  for (let x = 0; x < width; x++) {
    let sum = 0;
    const start = x * step;
    for (let i = 0; i < step && start + i < data.length; i++) {
      sum += data[start + i];
    }
    ctx.lineTo(x, mid + (sum / step) * amp);
  }
  ctx.stroke();
}

export function WaveformHeatmap({ audioUrl, windows }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [durationMs, setDurationMs] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);

  const reconMin = windows.length
    ? Math.min(...windows.map((w) => w.recon_score))
    : 0;
  const reconMax = windows.length
    ? Math.max(...windows.map((w) => w.recon_score))
    : 1;

  useEffect(() => {
    if (!audioUrl || !canvasRef.current) return;

    let cancelled = false;
    setLoadError(null);

    const load = async () => {
      try {
        const res = await fetch(audioUrl);
        const buf = await res.arrayBuffer();
        const ctx = new AudioContext();
        const audioBuffer = await ctx.decodeAudioData(buf.slice(0));
        await ctx.close();

        if (cancelled || !canvasRef.current) return;

        setDurationMs(audioBuffer.duration * 1000);
        drawWaveform(canvasRef.current, audioBuffer);
      } catch (e) {
        if (!cancelled) {
          setLoadError(e instanceof Error ? e.message : "Failed to decode audio");
        }
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [audioUrl]);

  useEffect(() => {
    const onResize = () => {
      if (!audioUrl || !canvasRef.current) return;
      fetch(audioUrl)
        .then((r) => r.arrayBuffer())
        .then((buf) => {
          const ctx = new AudioContext();
          return ctx.decodeAudioData(buf.slice(0)).then((ab) => {
            ctx.close();
            if (canvasRef.current) drawWaveform(canvasRef.current, ab);
          });
        })
        .catch(() => {});
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [audioUrl]);

  const timelineEnd =
    durationMs > 0
      ? durationMs
      : windows.length
        ? Math.max(...windows.map((w) => w.end_ms))
        : 1;

  return (
    <div className="waveform-card">
      <div className="waveform-header">
        <h2>Waveform &amp; reconstruction heatmap</h2>
        <div className="legend">
          <span className="legend-swatch low" />
          <span>Low recon error</span>
          <span className="legend-swatch high" />
          <span>High recon error</span>
        </div>
      </div>

      <div className="waveform-stack">
        <canvas ref={canvasRef} className="waveform-canvas" />
        <div ref={overlayRef} className="heatmap-overlay" aria-hidden>
          {windows.map((w, i) => {
            const left = (w.start_ms / timelineEnd) * 100;
            const width = ((w.end_ms - w.start_ms) / timelineEnd) * 100;
            return (
              <div
                key={`${w.start_ms}-${i}`}
                className="heat-region"
                style={{
                  left: `${left}%`,
                  width: `${Math.max(width, 0.5)}%`,
                  backgroundColor: reconHeatColor(
                    w.recon_score,
                    reconMin,
                    reconMax,
                  ),
                  borderBottomColor: reconBorderColor(
                    w.recon_score,
                    reconMin,
                    reconMax,
                  ),
                }}
                title={`${w.start_ms.toFixed(0)}–${w.end_ms.toFixed(0)} ms | recon: ${w.recon_score.toFixed(4)} | token: ${w.token_score.toFixed(4)}`}
              />
            );
          })}
        </div>
      </div>

      {loadError && <p className="waveform-error">{loadError}</p>}

      {windows.length > 0 && (
        <div className="window-table-wrap">
          <table className="window-table">
            <thead>
              <tr>
                <th>Start (ms)</th>
                <th>End (ms)</th>
                <th>Recon score</th>
                <th>Token score</th>
              </tr>
            </thead>
            <tbody>
              {windows.map((w, i) => (
                <tr key={i}>
                  <td>{w.start_ms}</td>
                  <td>{w.end_ms}</td>
                  <td>{w.recon_score}</td>
                  <td>{w.token_score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
