import { useState } from 'react'
import { Users, Zap, RefreshCw } from 'lucide-react'
import { useApi, apiFetch } from '../hooks/useApi'
import TeamBadge from '../components/TeamBadge'
import StatusBadge from '../components/StatusBadge'

interface Team { tri: string; name: string; color: string }

interface InjuryRow {
  player_name: string; status: string; status_rank: number
}

interface RawPlayer {
  personId: number; name: string; position?: string
  predictions: Record<string, number>
  season_avg: Record<string, number>
  role?: 'starter' | 'bench'
}

interface OptResult {
  player: string
  position: string
  pts_avg: number
  reb_avg: number
  ast_avg: number
  min_pred: number
  score: number
  role?: 'starter' | 'bench'
}

const POS_COLORS: Record<string, { bg: string; text: string }> = {
  G: { bg: '#dcfce7', text: '#15803d' },
  F: { bg: '#ede9fe', text: '#6d28d9' },
  C: { bg: '#fef3c7', text: '#d97706' },
}

function PosBadge({ pos }: { pos: string }) {
  const primary = pos.charAt(0)
  const clr = POS_COLORS[primary] ?? { bg: '#f1f5f9', text: '#64748b' }
  return (
    <span style={{ fontSize: 10, fontWeight: 800, padding: '2px 7px', borderRadius: 4, background: clr.bg, color: clr.text, letterSpacing: '0.04em' }}>
      {pos}
    </span>
  )
}

function ScoreBar({ val, max }: { val: number; max: number }) {
  const pct = max > 0 ? Math.min(100, (val / max) * 100) : 0
  return (
    <div style={{ width: 80, height: 5, background: '#f1f5f9', borderRadius: 3, overflow: 'hidden' }}>
      <div style={{ width: `${pct}%`, height: '100%', background: 'linear-gradient(90deg, #4f46e5, #7c3aed)', borderRadius: 3 }} />
    </div>
  )
}

function scorePlayer(p: RawPlayer): number {
  const pr = p.predictions
  return (pr.pts ?? 0) + (pr.reb ?? 0) * 1.2 + (pr.ast ?? 0) * 1.5 +
         (pr.stl ?? 0) * 2 + (pr.blk ?? 0) * 2 - (pr.tov ?? 0) * 1.5 +
         (pr.min ?? 0) * 0.3
}

function toOptResult(p: RawPlayer): OptResult {
  return {
    player:   p.name,
    position: p.position ?? 'F',
    pts_avg:  p.season_avg.pts ?? 0,
    reb_avg:  p.season_avg.reb ?? 0,
    ast_avg:  p.season_avg.ast ?? 0,
    min_pred: p.predictions.min ?? 0,
    score:    scorePlayer(p),
    role:     p.role,
  }
}

function buildLineup(players: RawPlayer[]): OptResult[] {
  const sortedAll = [...players].sort((a, b) => scorePlayer(b) - scorePlayer(a))
  const bucket = (pos: string, p: RawPlayer) => (p.position ?? 'F').charAt(0) === pos

  const guards  = sortedAll.filter(p => bucket('G', p))
  const forwards = sortedAll.filter(p => bucket('F', p))
  const centers = sortedAll.filter(p => bucket('C', p))

  const selected = new Set<string>()
  const lineup: RawPlayer[] = []

  const pick = (pool: RawPlayer[], n: number) => {
    let added = 0
    for (const p of pool) {
      if (added >= n || lineup.length >= 5) break
      if (!selected.has(p.name)) { selected.add(p.name); lineup.push(p); added++ }
    }
  }

  pick(guards, 2)
  pick(forwards, 2)
  pick(centers, 1)

  // Backfill if any position bucket was empty
  for (const p of sortedAll) {
    if (lineup.length >= 5) break
    if (!selected.has(p.name)) { selected.add(p.name); lineup.push(p) }
  }

  return lineup.sort((a, b) => scorePlayer(b) - scorePlayer(a)).map(toOptResult)
}

export default function LineupOptimizer() {
  const { data: teams } = useApi<Team[]>('/teams')
  const [myTeam, setMyTeam] = useState('')
  const [oppTeam, setOppTeam] = useState('')
  const [result, setResult] = useState<{ team: string; lineup: OptResult[]; total_score: number } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const { data: myInj }  = useApi<{ injuries: InjuryRow[] }>(myTeam  ? `/injuries?team=${myTeam}`  : null, [myTeam])
  const { data: oppInj } = useApi<{ injuries: InjuryRow[] }>(oppTeam ? `/injuries?team=${oppTeam}` : null, [oppTeam])

  const myOut  = (myInj?.injuries  ?? []).filter(r => r.status_rank >= 4).map(r => r.player_name)
  const oppOut = (oppInj?.injuries ?? []).filter(r => r.status_rank >= 4).map(r => r.player_name)

  const run = async () => {
    if (!myTeam) return
    setLoading(true); setError('')
    try {
      const params = new URLSearchParams({ team: myTeam, top_n: '12' })
      if (oppTeam) params.set('opp', oppTeam)
      if (myOut.length) params.set('injured_out', myOut.join(','))

      const preds = await apiFetch<{ players: RawPlayer[] }>(`/predict/players?${params}`)
      const eligible = preds.players.filter(p => p.role === 'starter' || (p.predictions.min ?? 0) >= 15)
      const lineup = buildLineup(eligible)

      setResult({ team: myTeam, lineup, total_score: lineup.reduce((s, p) => s + p.score, 0) })
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const myTeamInfo  = teams?.find(t => t.tri === myTeam)
  const oppTeamInfo = teams?.find(t => t.tri === oppTeam)
  const maxScore = result ? Math.max(...result.lineup.map(p => p.score)) : 0
  const posSummary = result?.lineup.map(p => p.position).join(' · ') ?? ''

  return (
    <div className="fade-in">
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 26, fontWeight: 900, color: '#0f172a', letterSpacing: '-0.5px', marginBottom: 4 }}>
          Lineup <span className="gradient-text">Optimizer</span>
        </h1>
        <p style={{ fontSize: 13, color: '#94a3b8', fontWeight: 500 }}>
          Position-constrained optimal 5 (2G · 2F · 1C) · injury-adjusted · matchup-aware
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
        {/* My team */}
        <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 12, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.05)' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: '#4f46e5', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Your Team
          </div>
          <select value={myTeam} onChange={e => { setMyTeam(e.target.value); setResult(null) }}
            style={{
              width: '100%', background: '#f8fafc', border: '1px solid #e2e8f0',
              borderRadius: 8, padding: '10px 12px', fontSize: 13,
              color: myTeam ? '#0f172a' : '#94a3b8', outline: 'none', marginBottom: 12,
            }}>
            <option value="">Select your team…</option>
            {teams?.map(t => <option key={t.tri} value={t.tri}>{t.name}</option>)}
          </select>

          {myTeamInfo && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <TeamBadge tri={myTeamInfo.tri} color={myTeamInfo.color} size={32} />
              <span style={{ fontWeight: 700, color: '#0f172a' }}>{myTeamInfo.name}</span>
            </div>
          )}

          {myOut.length > 0 && (
            <div style={{ fontSize: 11, color: '#d97706', background: '#fef3c7', border: '1px solid #fde68a', borderRadius: 6, padding: '6px 10px', fontWeight: 600 }}>
              {myOut.length} player{myOut.length > 1 ? 's' : ''} out: {myOut.slice(0, 3).join(', ')}{myOut.length > 3 ? ` +${myOut.length - 3}` : ''}
            </div>
          )}

          {myInj?.injuries.filter(r => r.status_rank >= 2).slice(0, 4).map((r, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6, fontSize: 12 }}>
              <StatusBadge status={r.status} small />
              <span style={{ color: '#475569' }}>{r.player_name}</span>
            </div>
          ))}
        </div>

        {/* Opponent */}
        <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 12, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.05)' }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Opponent (optional)
          </div>
          <select value={oppTeam} onChange={e => { setOppTeam(e.target.value); setResult(null) }}
            style={{
              width: '100%', background: '#f8fafc', border: '1px solid #e2e8f0',
              borderRadius: 8, padding: '10px 12px', fontSize: 13,
              color: oppTeam ? '#0f172a' : '#94a3b8', outline: 'none', marginBottom: 12,
            }}>
            <option value="">Any opponent (generic)</option>
            {teams?.filter(t => t.tri !== myTeam).map(t => <option key={t.tri} value={t.tri}>{t.name}</option>)}
          </select>

          {oppTeamInfo && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <TeamBadge tri={oppTeamInfo.tri} color={oppTeamInfo.color} size={32} />
              <span style={{ fontWeight: 700, color: '#0f172a' }}>{oppTeamInfo.name}</span>
            </div>
          )}

          {oppOut.length > 0 && (
            <div style={{ fontSize: 11, color: '#16a34a', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 6, padding: '6px 10px', fontWeight: 600 }}>
              Opponent missing: {oppOut.slice(0, 3).join(', ')}
            </div>
          )}

          {oppInj?.injuries.filter(r => r.status_rank >= 4).slice(0, 3).map((r, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6, fontSize: 12 }}>
              <StatusBadge status={r.status} small />
              <span style={{ color: '#475569' }}>{r.player_name}</span>
            </div>
          ))}
        </div>
      </div>

      <button onClick={run} disabled={!myTeam || loading}
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '12px 28px', borderRadius: 10, fontSize: 14, fontWeight: 700,
          border: 'none', cursor: myTeam && !loading ? 'pointer' : 'not-allowed',
          background: myTeam ? 'linear-gradient(135deg, #4f46e5, #7c3aed)' : '#f1f5f9',
          color: myTeam ? '#fff' : '#94a3b8',
          marginBottom: 24, transition: 'all 0.2s',
          boxShadow: myTeam ? '0 4px 16px rgba(79,70,229,0.3)' : 'none',
        }}>
        {loading ? <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <Zap size={16} />}
        {loading ? 'Optimizing…' : 'Generate Optimal Lineup'}
      </button>

      {error && (
        <div style={{ color: '#dc2626', fontSize: 13, marginBottom: 16, padding: '10px 16px', background: '#fee2e2', borderRadius: 8, border: '1px solid #fca5a5', fontWeight: 600 }}>
          {error}
        </div>
      )}

      {result && (
        <div className="fade-in" style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 12, overflow: 'hidden', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
          <div style={{ padding: '14px 20px', borderBottom: '1px solid #f1f5f9', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#f8fafc' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Users size={16} color="#4f46e5" />
              <span style={{ fontWeight: 800, color: '#0f172a', fontSize: 14 }}>Optimal Starting 5</span>
              {oppTeamInfo && <span style={{ fontSize: 11, color: '#94a3b8' }}>vs {oppTeamInfo.name}</span>}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ fontSize: 11, color: '#94a3b8' }}>{posSummary}</span>
              <span style={{ fontSize: 11, color: '#64748b', fontWeight: 600 }}>score: <span style={{ color: '#4f46e5' }}>{result.total_score.toFixed(1)}</span></span>
            </div>
          </div>

          <div>
            {result.lineup.map((p, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '14px 20px',
                borderBottom: i < 4 ? '1px solid #f8fafc' : 'none',
                transition: 'background 0.1s',
              }}
                onMouseEnter={e => (e.currentTarget.style.background = '#fafbff')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>

                <span style={{
                  width: 26, height: 26, borderRadius: '50%', flexShrink: 0,
                  background: i === 0 ? 'linear-gradient(135deg, #4f46e5, #7c3aed)' : '#f1f5f9',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 12, fontWeight: 800, color: i === 0 ? '#fff' : '#94a3b8',
                }}>{i + 1}</span>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontWeight: 700, fontSize: 14, color: '#0f172a', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{p.player}</span>
                    <PosBadge pos={p.position} />
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 16, fontSize: 12, flexShrink: 0 }}>
                  {[
                    { val: p.pts_avg, label: 'PTS', color: '#4f46e5' },
                    { val: p.reb_avg, label: 'REB', color: '#475569' },
                    { val: p.ast_avg, label: 'AST', color: '#475569' },
                    { val: p.min_pred, label: 'MIN', color: '#475569' },
                  ].map(({ val, label, color }) => (
                    <div key={label} style={{ textAlign: 'center', minWidth: 32 }}>
                      <div style={{ fontWeight: 700, color }}>{val.toFixed(1)}</div>
                      <div style={{ color: '#cbd5e1', fontSize: 10, fontWeight: 600 }}>{label}</div>
                    </div>
                  ))}
                </div>

                <ScoreBar val={p.score} max={maxScore} />
                <span style={{ fontSize: 12, fontWeight: 800, color: '#4f46e5', width: 40, textAlign: 'right', flexShrink: 0 }}>
                  {p.score.toFixed(1)}
                </span>
              </div>
            ))}
          </div>

          <div style={{ padding: '10px 20px', borderTop: '1px solid #f1f5f9', background: '#f8fafc' }}>
            <p style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.6 }}>
              Score = pts + 1.2×reb + 1.5×ast + 2×stl + 2×blk − 1.5×tov + 0.3×min · 2G+2F+1C constraint · next-game predictions{oppTeam ? ` vs ${oppTeamInfo?.name}` : ''}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
