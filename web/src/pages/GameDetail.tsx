import { useParams, useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { ArrowLeft, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { apiFetch } from '../hooks/useApi'
import TeamBadge from '../components/TeamBadge'
import StatusBadge from '../components/StatusBadge'
import Skeleton from '../components/Skeleton'

interface PlayerPred {
  personId: number; name: string; team: string
  position?: string
  predictions: Record<string, number>
  season_avg: Record<string, number>
  role?: 'starter' | 'bench'
}

interface InjuryRow {
  team_tri: string; player_name: string
  position: string; status: string
  status_rank: number; injury_detail: string
}

interface GamePred {
  home: string; away: string
  home_name: string; away_name: string
  home_color: string; away_color: string
  home_logo?: string; away_logo?: string
  win_prob_home: number; win_prob_away: number
  home_score: number; away_score: number
  model_acc: number; model_auc: number
}

interface TeamPreds {
  players: PlayerPred[]
  context: { opp: string; opp_def_rating: number | null; min_redistributed: number; injured_out: string | null }
}

const STATS = ['pts','reb','ast','stl','blk','tov','min']

const POS_COLORS: Record<string, { bg: string; text: string }> = {
  G: { bg: '#dcfce7', text: '#15803d' },
  F: { bg: '#ede9fe', text: '#6d28d9' },
  C: { bg: '#fef3c7', text: '#d97706' },
}

function Trend({ pred, avg }: { pred: number; avg: number }) {
  const diff = pred - avg
  if (Math.abs(diff) < 1.5) return <Minus size={12} color="#94a3b8" />
  if (diff > 0) return <TrendingUp size={12} color="#16a34a" />
  return <TrendingDown size={12} color="#dc2626" />
}

function StatBar({ label, pred, avg }: { label: string; pred: number; avg: number }) {
  const pct = avg > 0 ? Math.min(100, (pred / avg) * 100) : 50
  const isUp = pred >= avg * 0.95
  const color = isUp ? '#4f46e5' : '#dc2626'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ fontSize: 10, color: '#94a3b8', width: 26, textTransform: 'uppercase', fontWeight: 600 }}>{label}</span>
      <div style={{ flex: 1, height: 4, background: '#f1f5f9', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 2, transition: 'width 0.5s ease' }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 700, color, width: 28, textAlign: 'right' }}>{pred}</span>
      <Trend pred={pred} avg={avg} />
    </div>
  )
}

function PlayerCard({ p, injured }: { p: PlayerPred; injured: boolean }) {
  const [open, setOpen] = useState(false)
  const isStarter = p.role === 'starter'
  const posClr = POS_COLORS[(p.position ?? 'F').charAt(0)] ?? POS_COLORS['F']

  return (
    <div style={{
      background: injured ? '#fafafa' : '#ffffff',
      border: `1px solid ${injured ? '#f1f5f9' : '#e2e8f0'}`,
      borderRadius: 10,
      padding: 12,
      opacity: injured ? 0.55 : 1,
      transition: 'all 0.2s',
      boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: open ? 10 : 0, cursor: 'pointer' }}
        onClick={() => setOpen(o => !o)}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <span style={{ fontWeight: 700, fontSize: 13, color: injured ? '#94a3b8' : '#0f172a' }}>{p.name}</span>
            {p.position && (
              <span style={{ fontSize: 9, fontWeight: 800, padding: '1px 5px', borderRadius: 3, background: posClr.bg, color: posClr.text, letterSpacing: '0.04em' }}>
                {p.position}
              </span>
            )}
            {p.role && (
              <span style={{
                fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 3,
                background: isStarter ? '#ede9fe' : '#f1f5f9',
                color: isStarter ? '#6d28d9' : '#64748b',
                letterSpacing: '0.04em',
              }}>{isStarter ? 'STARTER' : 'BENCH'}</span>
            )}
            {injured && (
              <span style={{ fontSize: 10, color: '#dc2626', background: '#fee2e2', padding: '1px 6px', borderRadius: 4, fontWeight: 700 }}>OUT</span>
            )}
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <span style={{ fontSize: 12, color: '#4f46e5', fontWeight: 700 }}>{p.predictions.pts} pts</span>
            <span style={{ fontSize: 12, color: '#64748b' }}>{p.predictions.reb} reb</span>
            <span style={{ fontSize: 12, color: '#64748b' }}>{p.predictions.ast} ast</span>
            <span style={{ fontSize: 12, color: '#64748b' }}>{p.predictions.min} min</span>
          </div>
        </div>
        <span style={{ fontSize: 10, color: '#cbd5e1', fontWeight: 700 }}>{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, borderTop: '1px solid #f1f5f9', paddingTop: 10 }}>
          {STATS.map(s => (
            <StatBar key={s} label={s} pred={p.predictions[s] ?? 0} avg={p.season_avg[s] ?? 0} />
          ))}
        </div>
      )}
    </div>
  )
}

function WinProbBar({ pred }: { pred: GamePred }) {
  const homeW = Math.round(pred.win_prob_home * 100)
  const awayW = 100 - homeW
  return (
    <div style={{
      background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 16,
      padding: 24, marginBottom: 20, boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <TeamBadge tri={pred.home} color={pred.home_color} size={52} logo={pred.home_logo} />
          <div>
            <div style={{ fontWeight: 800, fontSize: 16, color: '#0f172a' }}>{pred.home_name}</div>
            <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>HOME</div>
          </div>
        </div>

        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 40, fontWeight: 900, letterSpacing: '-2px', lineHeight: 1 }}>
            <span style={{ color: pred.home_color }}>{Math.round(pred.home_score)}</span>
            <span style={{ color: '#e2e8f0', margin: '0 10px' }}>–</span>
            <span style={{ color: pred.away_color }}>{Math.round(pred.away_score)}</span>
          </div>
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>projected score</div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexDirection: 'row-reverse' }}>
          <TeamBadge tri={pred.away} color={pred.away_color} size={52} logo={pred.away_logo} />
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontWeight: 800, fontSize: 16, color: '#0f172a' }}>{pred.away_name}</div>
            <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>AWAY</div>
          </div>
        </div>
      </div>

      <div style={{ marginBottom: 6 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 13, fontWeight: 800 }}>
          <span style={{ color: pred.home_color }}>{homeW}%</span>
          <span style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>win probability</span>
          <span style={{ color: pred.away_color }}>{awayW}%</span>
        </div>
        <div style={{ height: 10, borderRadius: 6, background: `linear-gradient(90deg, ${pred.home_color} ${homeW}%, ${pred.away_color} ${homeW}%)`, boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.1)' }} />
      </div>

      <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginTop: 12 }}>
        <span style={{ fontSize: 11, color: '#94a3b8' }}>Model acc: <span style={{ color: '#4f46e5', fontWeight: 700 }}>{Math.round(pred.model_acc * 100)}%</span></span>
        <span style={{ fontSize: 11, color: '#94a3b8' }}>AUC: <span style={{ color: '#4f46e5', fontWeight: 700 }}>{pred.model_auc.toFixed(3)}</span></span>
      </div>
    </div>
  )
}

export default function GameDetail() {
  const { gameId } = useParams<{ gameId: string }>()
  const nav = useNavigate()

  const [loading, setLoading] = useState(true)
  const [pred, setPred] = useState<GamePred | null>(null)
  const [homeInj, setHomeInj] = useState<InjuryRow[]>([])
  const [awayInj, setAwayInj] = useState<InjuryRow[]>([])
  const [homePreds, setHomePreds] = useState<TeamPreds | null>(null)
  const [awayPreds, setAwayPreds] = useState<TeamPreds | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!gameId) return
    const parts = gameId.replace(/-vs-/i, '|').split('|')
    if (parts.length !== 2) { setError('Invalid game ID'); setLoading(false); return }
    const [home, away] = [parts[0].toUpperCase(), parts[1].toUpperCase()]

    const load = async () => {
      try {
        const [gamePred, homeInjData, awayInjData] = await Promise.all([
          apiFetch<GamePred>(`/predict/game?home=${home}&away=${away}`),
          apiFetch<{ injuries: InjuryRow[] }>(`/injuries?team=${home}`),
          apiFetch<{ injuries: InjuryRow[] }>(`/injuries?team=${away}`),
        ])
        setPred(gamePred)
        setHomeInj(homeInjData.injuries)
        setAwayInj(awayInjData.injuries)

        const homeOut = homeInjData.injuries.filter(r => r.status_rank >= 4).map(r => r.player_name).join(',')
        const awayOut = awayInjData.injuries.filter(r => r.status_rank >= 4).map(r => r.player_name).join(',')

        const [hp, ap] = await Promise.all([
          apiFetch<TeamPreds>(`/predict/players?team=${home}&top_n=8&opp=${away}&is_home=true${homeOut ? `&injured_out=${encodeURIComponent(homeOut)}` : ''}`),
          apiFetch<TeamPreds>(`/predict/players?team=${away}&top_n=8&opp=${home}&is_home=false${awayOut ? `&injured_out=${encodeURIComponent(awayOut)}` : ''}`),
        ])
        setHomePreds(hp)
        setAwayPreds(ap)
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [gameId])

  if (loading) return (
    <div className="fade-in">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Skeleton height={160} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[...Array(5)].map((_, i) => <div key={i} className="shimmer" style={{ height: 64, borderRadius: 10 }} />)}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[...Array(5)].map((_, i) => <div key={i} className="shimmer" style={{ height: 64, borderRadius: 10 }} />)}
          </div>
        </div>
      </div>
    </div>
  )

  if (error) return (
    <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 12, padding: 40, textAlign: 'center', color: '#dc2626', boxShadow: '0 1px 4px rgba(0,0,0,0.05)' }}>
      {error}
    </div>
  )

  const homeOutNames = new Set(homeInj.filter(r => r.status_rank >= 4).map(r => r.player_name.toLowerCase()))
  const awayOutNames = new Set(awayInj.filter(r => r.status_rank >= 4).map(r => r.player_name.toLowerCase()))

  const renderInjuries = (injs: InjuryRow[]) => {
    const notable = injs.filter(r => r.status_rank >= 2)
    if (!notable.length) return <div style={{ fontSize: 12, color: '#94a3b8', padding: '6px 0' }}>No notable injuries</div>
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        {notable.map((r, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
            <StatusBadge status={r.status} small />
            <span style={{ color: '#0f172a', fontWeight: 600 }}>{r.player_name}</span>
            {r.injury_detail && <span style={{ color: '#94a3b8' }}>· {r.injury_detail}</span>}
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="fade-in">
      <button onClick={() => nav(-1)} style={{
        display: 'flex', alignItems: 'center', gap: 6, marginBottom: 20,
        background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: 13, fontWeight: 600,
        padding: '6px 0',
      }}>
        <ArrowLeft size={14} /> Back
      </button>

      {pred && <WinProbBar pred={pred} />}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Home */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
            {pred && <TeamBadge tri={pred.home} color={pred.home_color} size={28} logo={pred.home_logo} />}
            <span style={{ fontWeight: 800, fontSize: 15, color: '#0f172a' }}>{pred?.home_name ?? 'Home'}</span>
          </div>

          <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 10, padding: 14, marginBottom: 10, boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Injury Report</div>
            {renderInjuries(homeInj)}
          </div>

          {(homePreds?.context.min_redistributed ?? 0) > 0 && (
            <div style={{ fontSize: 11, color: '#d97706', background: '#fef3c7', border: '1px solid #fde68a', borderRadius: 6, padding: '6px 10px', marginBottom: 10, fontWeight: 600 }}>
              +{homePreds!.context.min_redistributed.toFixed(0)} min redistributed from injured players
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {(homePreds?.players ?? []).map(p => (
              <PlayerCard key={p.personId} p={p} injured={homeOutNames.has(p.name.toLowerCase())} />
            ))}
          </div>
        </div>

        {/* Away */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
            {pred && <TeamBadge tri={pred.away} color={pred.away_color} size={28} logo={pred.away_logo} />}
            <span style={{ fontWeight: 800, fontSize: 15, color: '#0f172a' }}>{pred?.away_name ?? 'Away'}</span>
          </div>

          <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 10, padding: 14, marginBottom: 10, boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Injury Report</div>
            {renderInjuries(awayInj)}
          </div>

          {(awayPreds?.context.min_redistributed ?? 0) > 0 && (
            <div style={{ fontSize: 11, color: '#d97706', background: '#fef3c7', border: '1px solid #fde68a', borderRadius: 6, padding: '6px 10px', marginBottom: 10, fontWeight: 600 }}>
              +{awayPreds!.context.min_redistributed.toFixed(0)} min redistributed from injured players
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {(awayPreds?.players ?? []).map(p => (
              <PlayerCard key={p.personId} p={p} injured={awayOutNames.has(p.name.toLowerCase())} />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
