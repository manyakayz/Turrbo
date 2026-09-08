import { useState, useEffect, useCallback } from 'react';
import {
  RefreshCw, AlertTriangle, Activity, Cpu, Wrench, ShieldAlert,
} from 'lucide-react';
import {
  Area, CartesianGrid, ComposedChart, Line,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

// ─── Types ────────────────────────────────────────────────────────────────────
// These mirror backend/app/main.py's EngineSummary / telemetry response shapes
// exactly. risk_level / maintenance_category / recommended_action are pure
// post-hoc business logic computed from (RUL, uncertainty) — see
// src/turbofan_rul/inference.py — not additional model outputs.

interface Engine {
  id: string;
  original_unit: number;
  cycle: number;
  rul_predicted: number;
  rul_std: number;
  status: 'Healthy' | 'Moderate' | 'Warning' | 'Critical';
  risk_level: 'Low' | 'Medium' | 'High' | 'Severe';
  maintenance_category: 'Healthy' | 'Inspection Recommended' | 'Maintenance Required' | 'Critical';
  recommended_action: string;
  uncertainty_elevated: boolean;
}

interface TelemetryPoint {
  cycle: number;
  t24: number;
  t30: number;
}

interface DegradationPoint {
  cycle: number;
  predicted: number;
  upper: number;
  lower: number;
  rollingAvg?: number | null;
}

interface OpsNote {
  kind: 'info' | 'warn' | 'crit';
  text: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Convert "ENG-10001" → "N-1001·A" style realistic fleet tail number */
const toFleetId = (raw: string): string => {
  const num = parseInt(raw.replace('ENG-', ''), 10);
  if (num >= 30001) return `CFM56-${num - 30000}·D`;
  if (num >= 20001) return `CFM56-${num - 20000}·C`;
  if (num >= 10001) return `V2500-${num - 10000}·B`;
  return `GE90-${num}·A`;
};

const statusDotClass = (status: string): string => {
  switch (status) {
    case 'Healthy':  return 'status-dot healthy';
    case 'Moderate': return 'status-dot moderate';
    case 'Warning':  return 'status-dot warning';
    case 'Critical': return 'status-dot critical';
    default: return 'status-dot';
  }
};

const badgeClass = (status: string): string =>
  `badge ${status.toLowerCase()}`;

const riskBadgeClass = (risk: string): string =>
  `badge ${risk.toLowerCase()}`;

const maintBadgeClass = (category: string): string => {
  switch (category) {
    case 'Healthy':                  return 'badge healthy-maint';
    case 'Inspection Recommended':   return 'badge inspection';
    case 'Maintenance Required':     return 'badge maintenance';
    case 'Critical':                 return 'badge critical-maint';
    default: return 'badge';
  }
};

/** Muted industrial chart color palette */
const CHART = {
  predicted:  '#4472a8',
  band:       '#b8ccdf',
  t24:        '#5a7fa0',
  t30:        '#7c9b6e',
  threshold:  '#a84040',
  warning:    '#c48a30',
  grid:       '#d5d9e0',
  axis:       '#8b94a6',
  rollingAvg: '#8ba5c4',
};

/**
 * Operational notes — every line here is derived from a real backend value
 * (maintenance recommendation, uncertainty, record count). Nothing here is
 * simulated/fabricated flavor text.
 */
const deriveOpsNotes = (engine: Engine, telLen: number): OpsNote[] => {
  const notes: OpsNote[] = [];
  const kind: OpsNote['kind'] =
    engine.maintenance_category === 'Critical' ? 'crit' :
    engine.maintenance_category === 'Maintenance Required' ? 'warn' : 'info';
  notes.push({ kind, text: `${engine.maintenance_category} — ${engine.recommended_action}` });
  if (engine.uncertainty_elevated) {
    notes.push({
      kind: 'warn',
      text: `Prediction uncertainty (±${engine.rul_std} cycles) pushed this recommendation up from the raw RUL zone`,
    });
  }
  if (telLen > 0) {
    notes.push({ kind: 'info', text: `${telLen} telemetry records analyzed this session` });
  }
  return notes;
};

/** Simple rolling-average smoother over real predicted values (not fabricated data) */
const rollingAvg = (arr: number[], w = 5): (number | null)[] =>
  arr.map((_, i) => {
    if (i < w - 1) return null;
    const slice = arr.slice(i - w + 1, i + 1);
    return slice.reduce((s, v) => s + v, 0) / w;
  });

/** Format NOW timestamp (real client-side fetch time, not simulated) */
const nowTs = (): string => {
  const d = new Date();
  return `${d.toISOString().split('T')[0]}  ${d.toTimeString().slice(0, 8)} UTC`;
};

/** Fleet health summary, derived client-side from the already-fetched engine list
 *  (rather than a second API call) so sidebar counts always match the badges
 *  shown per-engine — a second MC-Dropout call would be stochastic and could
 *  disagree slightly with what's on screen. */
const fleetHealth = (engines: Engine[]): { ok: number; warn: number; crit: number } => ({
  ok:   engines.filter(e => e.status === 'Healthy' || e.status === 'Moderate').length,
  warn: engines.filter(e => e.status === 'Warning').length,
  crit: engines.filter(e => e.status === 'Critical').length,
});

const fleetRiskBreakdown = (engines: Engine[]): Record<string, number> => {
  const out: Record<string, number> = {};
  for (const e of engines) out[e.risk_level] = (out[e.risk_level] || 0) + 1;
  return out;
};

function App() {
  const [engines, setEngines]           = useState<Engine[]>([]);
  const [selectedEngine, setSelected]   = useState<Engine | null>(null);
  const [telemetryData, setTelemetry]   = useState<TelemetryPoint[]>([]);
  const [degradationData, setDegrad]    = useState<DegradationPoint[]>([]);
  const [opsNotes, setOpsNotes]         = useState<OpsNote[]>([]);
  const [lastUpdated, setLastUpdated]   = useState<string>('—');
  const [isLoadingFleet, setFleetLoad]  = useState(true);
  const [isEngineLoading, setEngLoad]   = useState(false);

  const handleSelectEngine = useCallback((engine: Engine) => {
    setSelected(engine);
    setEngLoad(true);
    setOpsNotes(deriveOpsNotes(engine, 0));

    fetch(`/api/engines/${engine.original_unit}/telemetry`)
      .then(res => res.json())
      .then(data => {
        const rawTel: TelemetryPoint[] = data.telemetry || [];

        // Enrich degradation with a rolling average of the real predicted values
        const rawDeg: DegradationPoint[] = data.degradation || [];
        const preds = rawDeg.map(d => d.predicted);
        const avg   = rollingAvg(preds, 4);
        const enrichedDeg = rawDeg.map((d, i) => ({ ...d, rollingAvg: avg[i] }));

        setTelemetry(rawTel);
        setDegrad(enrichedDeg);
        setOpsNotes(deriveOpsNotes(engine, rawTel.length));
        setLastUpdated(nowTs());
        setEngLoad(false);
      })
      .catch(err => {
        console.error('Telemetry fetch failed:', err);
        setEngLoad(false);
      });
  }, []);

  useEffect(() => {
    fetch('/api/engines')
      .then(res => res.json())
      .then((data: Engine[]) => {
        setEngines(data);
        if (data.length > 0) handleSelectEngine(data[0]);
        setFleetLoad(false);
      })
      .catch(() => setFleetLoad(false));
  }, [handleSelectEngine]);


  const fh = fleetHealth(engines);
  const riskBreakdown = fleetRiskBreakdown(engines);
  const ciLower = selectedEngine ? Math.max(0, Math.round(selectedEngine.rul_predicted - 2 * selectedEngine.rul_std)) : 0;
  const ciUpper = selectedEngine ? Math.round(selectedEngine.rul_predicted + 2 * selectedEngine.rul_std) : 0;

  return (
    <div className="app-shell" style={{ height: '100vh', width: '100%', background: 'var(--bg-base)', color: 'var(--text-primary)', display: 'flex', overflow: 'hidden', fontFamily: 'var(--font-body)', position: 'relative' }}>

      {/* ── LEFT SIDEBAR ── Fleet Registry */}
      <aside className="app-sidebar" style={{ width: 220, background: 'var(--bg-panel)', borderRight: '1px solid var(--border-faint)', display: 'flex', flexDirection: 'column', flexShrink: 0, zIndex: 10, overflow: 'hidden' }}>

        {/* Wordmark */}
        <div style={{ padding: '10px 14px 8px', borderBottom: '1px solid var(--border-sep)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <Activity size={13} color="var(--accent-blue)" />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '0.04em' }}>FLEET OPS · PROGNOSTICS</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', marginTop: 2, letterSpacing: '0.06em' }}>CMAPSS TURBOFAN MONITOR</div>
        </div>

        {/* Fleet Health Summary */}
        <div style={{ padding: '8px 14px', borderBottom: '1px solid var(--border-sep)' }}>
          <div className="label-xs" style={{ marginBottom: 4 }}>Fleet health</div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--status-healthy)' }}>{fh.ok}<span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> ok</span></span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--amber)' }}>{fh.warn}<span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> warn</span></span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--red)' }}>{fh.crit}<span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> crit</span></span>
          </div>
          <div className="label-xs" style={{ marginBottom: 4 }}>Risk level</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-tertiary)' }}>
            {(['Low', 'Medium', 'High', 'Severe'] as const).map((r, i) => (
              <span key={r}>{i > 0 ? ' · ' : ''}{riskBreakdown[r] || 0} {r.toLowerCase()}</span>
            ))}
          </div>
        </div>

        {/* Section label */}
        <div style={{ padding: '7px 14px 4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span className="label-xs">Active units</span>
          {isLoadingFleet && <RefreshCw size={10} className="spin" color="var(--text-muted)" />}
        </div>

        {/* Engine list */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {engines.map(eng => (
            <button
              key={eng.id}
              onClick={() => handleSelectEngine(eng)}
              className={`fleet-row${selectedEngine?.id === eng.id ? ' active' : ''}`}
              style={{ width: '100%', textAlign: 'left', background: 'none', border: 'none', cursor: 'pointer' }}
            >
              <div className={`${statusDotClass(eng.status)}${eng.status === 'Critical' ? ' blink' : ''}`} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 500, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {toFleetId(eng.id)}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', marginTop: 1 }}>
                  {eng.cycle} cyc · RUL {eng.rul_predicted}
                </div>
              </div>
              <span className={badgeClass(eng.status)} style={{ flexShrink: 0, fontSize: 9 }}>{eng.status.slice(0, 4).toUpperCase()}</span>
            </button>
          ))}
          {engines.length === 0 && !isLoadingFleet && (
            <div style={{ padding: '12px 14px', fontSize: 11, color: 'var(--text-muted)' }}>No units loaded</div>
          )}
        </div>

        {/* Footer — real, static facts about the deployed pipeline (see MODELS.md) */}
        <div style={{ padding: '8px 14px', borderTop: '1px solid var(--border-sep)' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', lineHeight: 1.6 }}>
            <div>BiLSTM v4 · Domain-Adapted</div>
            <div>MC Dropout · N=10 (fleet) / N=50 (detail)</div>
            <div style={{ color: 'var(--accent-blue)', marginTop: 2 }}>NASA C-MAPSS FD001–FD004</div>
          </div>
        </div>
      </aside>

      {/* ── MAIN AREA ── */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, overflow: 'hidden', background: 'var(--bg-base)' }}>

        {/* ── HEADER ── */}
        <header className="app-header" style={{ height: 40, background: 'var(--bg-panel)', borderBottom: '1px solid var(--border-faint)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 16px', flexShrink: 0, zIndex: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            {selectedEngine && (
              <>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                  {toFleetId(selectedEngine.id)}
                </span>
                <span style={{ color: 'var(--border-mid)', fontSize: 11 }}>·</span>
                <span className={badgeClass(selectedEngine.status)}>{selectedEngine.status.toUpperCase()}</span>
                <span style={{ color: 'var(--border-mid)', fontSize: 11 }}>·</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>cycle {selectedEngine.cycle}</span>
              </>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>Last refresh · {lastUpdated}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: isLoadingFleet ? 'var(--amber)' : 'var(--status-healthy)' }} className={isLoadingFleet ? 'blink' : ''} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>{isLoadingFleet ? 'INITIALIZING' : 'TELEMETRY LIVE'}</span>
            </div>
          </div>
        </header>

        {/* ── SCROLLABLE CONTENT ── */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px', position: 'relative' }}>
          {isEngineLoading && (
            <div style={{ position: 'absolute', inset: 0, background: 'rgba(237,238,240,0.7)', zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(2px)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-tertiary)' }}>
                <RefreshCw size={14} className="spin" /> Loading telemetry…
              </div>
            </div>
          )}

          {selectedEngine && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxWidth: 1400 }}>

              {/* ── ROW 1: Ops metrics strip (asymmetric) ── */}
              <div className="metrics-row" style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr 1.4fr', gap: 8 }}>

                {/* RUL — wide primary, now with explicit numeric confidence interval */}
                <div className="panel" style={{ padding: '10px 14px' }}>
                  <div className="label-xs" style={{ marginBottom: 6 }}>Remaining cycles · RUL</div>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 500, color: selectedEngine.status === 'Critical' ? 'var(--amber-hot)' : selectedEngine.status === 'Warning' ? 'var(--amber)' : selectedEngine.status === 'Moderate' ? 'var(--amber-dim)' : 'var(--text-primary)', lineHeight: 1 }}>
                      {selectedEngine.rul_predicted}
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>cycles</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 6 }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>±{selectedEngine.rul_std}σ · 95% CI [{ciLower}–{ciUpper}]</span>
                  </div>
                  <div className="progress-track" style={{ marginTop: 6 }}>
                    <div className="progress-fill" style={{ width: `${Math.min(100, (selectedEngine.rul_predicted / 150) * 100)}%`, background: selectedEngine.status === 'Critical' ? 'var(--red-dim)' : selectedEngine.status === 'Warning' ? 'var(--amber-dim)' : 'var(--accent-blue-dim)' }} />
                  </div>
                </div>

                {/* Cycle count */}
                <div className="panel" style={{ padding: '10px 14px' }}>
                  <div className="label-xs" style={{ marginBottom: 6 }}>Engine cycles</div>
                  <div className="value-lg">{selectedEngine.cycle}</div>
                  <div className="label-sm" style={{ marginTop: 4 }}>Total recorded</div>
                </div>

                {/* Risk level — combines RUL + uncertainty, see inference.py */}
                <div className="panel" style={{ padding: '10px 14px' }}>
                  <div className="label-xs" style={{ marginBottom: 6 }}>Risk level</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                    <ShieldAlert size={13} color="var(--accent-blue)" />
                    <span className={riskBadgeClass(selectedEngine.risk_level)}>{selectedEngine.risk_level.toUpperCase()}</span>
                  </div>
                  <div className="label-sm" style={{ marginTop: 4 }}>RUL + uncertainty combined</div>
                </div>

                {/* Maintenance recommendation category */}
                <div className="panel" style={{ padding: '10px 14px' }}>
                  <div className="label-xs" style={{ marginBottom: 6 }}>Maintenance</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                    <Wrench size={12} color="var(--accent-blue)" />
                    <span className={maintBadgeClass(selectedEngine.maintenance_category)} style={{ fontSize: 9 }}>{selectedEngine.maintenance_category.toUpperCase()}</span>
                  </div>
                  {selectedEngine.uncertainty_elevated && (
                    <div className="label-sm" style={{ marginTop: 4, color: 'var(--amber)' }}>Elevated by uncertainty</div>
                  )}
                </div>

                {/* Model info — real, static facts only */}
                <div className="panel" style={{ padding: '10px 14px' }}>
                  <div className="label-xs" style={{ marginBottom: 6 }}>Active model</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 6 }}>
                    <Cpu size={11} color="var(--accent-blue)" />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)', fontWeight: 500 }}>BiLSTM v4 · Domain-Adapted</span>
                  </div>
                  <div className="sep-h" style={{ margin: '6px 0' }} />
                  <div className="label-xs" style={{ marginBottom: 4 }}>Session telemetry</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>{telemetryData.length} records analyzed</div>
                </div>
              </div>

              {/* ── ROW 2: Main charts (asymmetric 3:2 split) ── */}
              <div className="charts-row" style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 8, minHeight: 360 }}>

                {/* Degradation / RUL Trajectory — dominant */}
                <div className="panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <div className="section-strip">
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '0.05em' }}>RUL TRAJECTORY · PROGNOSTIC CURVE</span>
                    <div style={{ marginLeft: 'auto', display: 'flex', gap: 12, alignItems: 'center' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                        <span style={{ display: 'inline-block', width: 18, height: 2, background: CHART.predicted }} />prediction
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                        <span style={{ display: 'inline-block', width: 18, height: 6, background: CHART.band, opacity: 0.5 }} />±2σ band
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                        <span style={{ display: 'inline-block', width: 18, height: 1, background: CHART.rollingAvg, borderTop: `1px dashed ${CHART.rollingAvg}` }} />4-cyc avg
                      </span>
                    </div>
                  </div>
                  <div style={{ flex: 1, padding: '8px 4px 4px 0' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={degradationData} margin={{ top: 8, right: 14, bottom: 4, left: -10 }}>
                        <defs>
                          <linearGradient id="rulGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={CHART.predicted} stopOpacity={0.18} />
                            <stop offset="100%" stopColor={CHART.predicted} stopOpacity={0.02} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="2 4" stroke={CHART.grid} vertical={false} />
                        <XAxis dataKey="cycle" stroke={CHART.axis} fontSize={9} tickLine={false} axisLine={{ stroke: CHART.grid }} tickMargin={6} fontFamily="'IBM Plex Mono',monospace" label={{ value: 'Engine cycle', position: 'insideBottom', offset: -2, fontSize: 9, fill: CHART.axis, fontFamily: "'IBM Plex Mono',monospace" }} />
                        <YAxis stroke={CHART.axis} fontSize={9} tickLine={false} axisLine={false} domain={[-10, 160]} tickMargin={6} fontFamily="'IBM Plex Mono',monospace" label={{ value: 'RUL (cycles)', angle: -90, position: 'insideLeft', offset: 12, fontSize: 9, fill: CHART.axis, fontFamily: "'IBM Plex Mono',monospace" }} />
                        <Tooltip
                          contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border-mid)', borderRadius: 2, fontSize: 10, fontFamily: "'IBM Plex Mono',monospace", padding: '6px 10px' }}
                          labelStyle={{ color: 'var(--text-muted)', marginBottom: 3 }}
                          itemStyle={{ color: 'var(--text-primary)' }}
                          formatter={(v) => [Number(v).toFixed(1), '']}
                        />
                        {/* Warning zone band */}
                        <ReferenceLine y={30} stroke={CHART.threshold} strokeDasharray="3 3" strokeWidth={1} label={{ position: 'insideTopRight', value: 'CRITICAL', fill: CHART.threshold, fontSize: 8, fontFamily: "'IBM Plex Mono',monospace" }} />
                        <ReferenceLine y={60} stroke={CHART.warning} strokeDasharray="2 4" strokeWidth={1} label={{ position: 'insideTopRight', value: 'WARNING', fill: CHART.warning, fontSize: 8, fontFamily: "'IBM Plex Mono',monospace" }} />
                        {/* Confidence band */}
                        <Area type="monotoneX" dataKey="upper" stroke="none" fill={CHART.band} fillOpacity={0.25} isAnimationActive={false} legendType="none" />
                        <Area type="monotoneX" dataKey="lower" stroke="none" fill="var(--bg-panel)" fillOpacity={1} isAnimationActive={false} legendType="none" />
                        {/* Main predicted curve */}
                        <Area type="monotoneX" dataKey="predicted" stroke={CHART.predicted} strokeWidth={2} fill="url(#rulGrad)" dot={false} activeDot={{ r: 3, fill: CHART.predicted, strokeWidth: 0 }} isAnimationActive={false} />
                        {/* Rolling average */}
                        <Line type="monotone" dataKey="rollingAvg" stroke={CHART.rollingAvg} strokeWidth={1} strokeDasharray="4 3" dot={false} isAnimationActive={false} connectNulls={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Sensor telemetry — secondary */}
                <div className="panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <div className="section-strip">
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '0.05em' }}>SENSOR TELEMETRY</span>
                    <div style={{ marginLeft: 'auto', display: 'flex', gap: 10, alignItems: 'center' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                        <span style={{ display: 'inline-block', width: 14, height: 2, background: CHART.t24 }} />T24
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                        <span style={{ display: 'inline-block', width: 14, height: 2, background: CHART.t30 }} />T30
                      </span>
                    </div>
                  </div>
                  <div style={{ flex: 1, padding: '8px 4px 4px 0' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={telemetryData} margin={{ top: 8, right: 14, bottom: 4, left: -10 }}>
                        <CartesianGrid strokeDasharray="2 4" stroke={CHART.grid} vertical={false} />
                        <XAxis dataKey="cycle" stroke={CHART.axis} fontSize={9} tickLine={false} axisLine={{ stroke: CHART.grid }} tickMargin={6} fontFamily="'IBM Plex Mono',monospace" />
                        <YAxis stroke={CHART.axis} fontSize={9} tickLine={false} axisLine={false} domain={['auto', 'auto']} tickMargin={6} fontFamily="'IBM Plex Mono',monospace" />
                        <Tooltip
                          contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border-mid)', borderRadius: 2, fontSize: 10, fontFamily: "'IBM Plex Mono',monospace", padding: '6px 10px' }}
                          labelStyle={{ color: 'var(--text-muted)', marginBottom: 3 }}
                          formatter={(v) => [Number(v).toFixed(4), '']}
                        />
                        <Line type="monotoneX" dataKey="t24" stroke={CHART.t24} strokeWidth={1.5} dot={false} activeDot={{ r: 3, fill: CHART.t24, strokeWidth: 0 }} isAnimationActive={false} />
                        <Line type="monotoneX" dataKey="t30" stroke={CHART.t30} strokeWidth={1.5} dot={false} activeDot={{ r: 3, fill: CHART.t30, strokeWidth: 0 }} isAnimationActive={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* ── ROW 3: Ops notes + alert log (asymmetric) ── */}
              <div className="notes-row" style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 8 }}>

                {/* Operational notes — derived entirely from real backend values */}
                <div className="panel" style={{ overflow: 'hidden' }}>
                  <div className="section-strip">
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '0.05em' }}>OPERATIONAL NOTES</span>
                  </div>
                  <div style={{ padding: '6px 0', display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {opsNotes.map((note, i) => (
                      <div key={i} className={`op-note${note.kind === 'warn' ? ' warn' : note.kind === 'crit' ? ' crit' : ''}`}>
                        {note.kind === 'crit' ? <AlertTriangle size={10} style={{ flexShrink: 0, marginTop: 1 }} /> :
                         note.kind === 'warn' ? <AlertTriangle size={10} style={{ flexShrink: 0, marginTop: 1 }} /> : null}
                        <span>{note.text}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Alert log */}
                <div className="panel" style={{ overflow: 'hidden' }}>
                  <div className="section-strip">
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '0.05em' }}>ALERT LOG · {toFleetId(selectedEngine.id)}</span>
                    <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>{lastUpdated}</span>
                  </div>
                  <div style={{ padding: '6px 12px', display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <div style={{ display: 'flex', gap: 10, padding: '5px 0', borderBottom: '1px solid var(--border-sep)', alignItems: 'flex-start' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 5, minWidth: 80, marginTop: 1 }}>
                        <div className={statusDotClass(selectedEngine.status)} />
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: selectedEngine.status === 'Critical' ? 'var(--amber-hot)' : selectedEngine.status === 'Warning' ? 'var(--amber)' : selectedEngine.status === 'Moderate' ? 'var(--amber-dim)' : 'var(--status-healthy)' }}>
                          {selectedEngine.status.toUpperCase()}
                        </span>
                      </div>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-tertiary)', lineHeight: 1.5 }}>
                        {toFleetId(selectedEngine.id)} — RUL projection {selectedEngine.rul_predicted} cycles · 95% CI [{ciLower}–{ciUpper}] · risk {selectedEngine.risk_level}
                      </span>
                    </div>
                    <div style={{ display: 'flex', gap: 10, padding: '5px 0', alignItems: 'flex-start' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 5, minWidth: 80, marginTop: 1 }}>
                        <div className={`status-dot${selectedEngine.maintenance_category === 'Critical' ? ' critical blink' : selectedEngine.maintenance_category === 'Maintenance Required' ? ' warning' : selectedEngine.maintenance_category === 'Inspection Recommended' ? ' moderate' : ' healthy'}`} />
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: 'var(--text-secondary)' }}>MAINT</span>
                      </div>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-tertiary)', lineHeight: 1.5 }}>
                        {selectedEngine.recommended_action}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

            </div>
          )}
        </div>
      </main>

    </div>
  );
}

export default App;
