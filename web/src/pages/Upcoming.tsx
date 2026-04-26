import { useNavigate } from 'react-router-dom'
import { useApi } from '../hooks/useApi'
import TeamBadge from '../components/TeamBadge'

interface UpcomingGame {
  game_id: string
  start_time: string
  home_tri: string; away_tri: string
  home_name: string; away_name: string
  home_color: string; away_color: string
  home_logo?: string; away_logo?: string
}

interface Pred {
  win_prob_home: number; win_prob_away: number
  home_score: number; away_score: number
}

function TeamBlock({ tri, name, color, logo, side }: { tri: string; name: string; color: string; logo?: string; side: 'home' | 'away' }) {
  const right = side === 'away'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexDirection: right ? 'row-reverse' : 'row' }}>
      <TeamBadge tri={tri} color={color} size={44} logo={logo} />
      <div style={{ textAlign: right ? 'right' : 'left' }}>
        <div style={{ fontWeight: 700, fontSize: 14, color: '#0f172a', lineHeight: 1.2 }}>{name}</div>
        <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2, fontWeight: 600, letterSpacing: '0.04em' }}>{right ? 'AWAY' : 'HOME'}</div>
      </div>
    </div>
  )
}

function GameRow({ g }: { g: UpcomingGame }) {
  const nav = useNavigate()
  const id = `${g.home_tri}-vs-${g.away_tri}`
  const { data: pred } = useApi<Pred>(`/predict/game?home=${g.home_tri}&away=${g.away_tri}`)

  const localTime = g.start_time
    ? new Date(g.start_time).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
    : ''

  const homeWin = pred ? pred.win_prob_home >= pred.win_prob_away : null
  const favColor = homeWin === null ? '#4f46e5' : homeWin ? g.home_color : g.away_color
  const favPct   = pred ? Math.round((homeWin ? pred.win_prob_home : pred.win_prob_away) * 100) : null

  return (
    <div onClick={() => nav(`/game/${id}`)} style={{
      background: '#ffffff',
      border: '1px solid #e2e8f0',
      borderRadius: 12, padding: '14px 20px',
      cursor: 'pointer', transition: 'all 0.18s ease',
      display: 'grid', gridTemplateColumns: '1fr auto 1fr',
      alignItems: 'center', gap: 16,
      boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
    }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLDivElement).style.boxShadow = '0 6px 20px rgba(0,0,0,0.08)'
        ;(e.currentTarget as HTMLDivElement).style.transform = 'translateY(-2px)'
        ;(e.currentTarget as HTMLDivElement).style.borderColor = '#a5b4fc'
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLDivElement).style.boxShadow = '0 1px 3px rgba(0,0,0,0.04)'
        ;(e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)'
        ;(e.currentTarget as HTMLDivElement).style.borderColor = '#e2e8f0'
      }}>

      <TeamBlock tri={g.home_tri} name={g.home_name} color={g.home_color} logo={g.home_logo} side="home" />

      <div style={{ textAlign: 'center', minWidth: 160 }}>
        {localTime && (
          <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {localTime}
          </div>
        )}
        {pred ? (
          <>
            <div style={{ fontSize: 26, fontWeight: 900, letterSpacing: '-1.5px', color: '#0f172a', lineHeight: 1 }}>
              <span style={{ color: g.home_color }}>{Math.round(pred.home_score)}</span>
              <span style={{ color: '#cbd5e1', margin: '0 8px' }}>–</span>
              <span style={{ color: g.away_color }}>{Math.round(pred.away_score)}</span>
            </div>
            <div style={{ fontSize: 10, color: '#94a3b8', margin: '4px 0 6px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>projected</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
              <div style={{ width: 72, height: 3, borderRadius: 2, overflow: 'hidden', background: '#e2e8f0' }}>
                <div style={{
                  height: '100%', width: `${Math.round(pred.win_prob_home * 100)}%`,
                  background: `linear-gradient(90deg, ${g.home_color}, ${g.away_color})`,
                }} />
              </div>
            </div>
            <div style={{ fontSize: 12, fontWeight: 700, color: favColor, marginTop: 4 }}>
              {homeWin ? g.home_tri : g.away_tri} · {favPct}%
            </div>
          </>
        ) : (
          <div className="shimmer" style={{ height: 56, borderRadius: 8, width: 140, margin: '0 auto' }} />
        )}
      </div>

      <TeamBlock tri={g.away_tri} name={g.away_name} color={g.away_color} logo={g.away_logo} side="away" />
    </div>
  )
}

function localDateKey(isoUtc: string): string {
  const d = new Date(isoUtc)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

function fmtDateKey(key: string): { label: string; sub: string } {
  const [y, m, day] = key.split('-').map(Number)
  const dt = new Date(y, m - 1, day)
  const todayMid = new Date(); todayMid.setHours(0, 0, 0, 0)
  const diff = Math.round((dt.getTime() - todayMid.getTime()) / 86400000)
  const dateStr = dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  if (diff === 0) return { label: 'Today', sub: dateStr }
  if (diff === 1) return { label: 'Tomorrow', sub: dateStr }
  return { label: dt.toLocaleDateString(undefined, { weekday: 'long' }), sub: dateStr }
}

export default function Upcoming() {
  const { data, loading } = useApi<{ games: UpcomingGame[] }>('/schedule/upcoming')

  // Group by device-local date, filter out past games
  const now = Date.now()
  const grouped = new Map<string, UpcomingGame[]>()
  for (const g of data?.games ?? []) {
    if (!g.start_time || new Date(g.start_time).getTime() < now) continue
    const key = localDateKey(g.start_time)
    if (!grouped.has(key)) grouped.set(key, [])
    grouped.get(key)!.push(g)
  }
  const days = [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b))

  return (
    <div className="fade-in" style={{ maxWidth: 900, margin: '0 auto' }}>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 28, fontWeight: 900, color: '#0f172a', letterSpacing: '-0.8px', marginBottom: 4 }}>
          Upcoming <span className="gradient-text">Schedule</span>
        </h1>
        <p style={{ fontSize: 12, color: '#94a3b8', fontWeight: 500 }}>
          Model predictions for next 7 days · 73% win accuracy · click for full game analysis
        </p>
      </div>

      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {[1, 2, 3].map(i => (
            <div key={i}>
              <div className="shimmer" style={{ width: 140, height: 14, borderRadius: 6, marginBottom: 12 }} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[1, 2].map(j => <div key={j} className="shimmer" style={{ height: 88, borderRadius: 12 }} />)}
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && days.length === 0 && (
        <div style={{ textAlign: 'center', padding: '60px 0', color: '#94a3b8', fontSize: 14 }}>
          No upcoming games in the next 7 days
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
        {days.map(([key, games]) => {
          const { label, sub } = fmtDateKey(key)
          return (
            <section key={key}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 10 }}>
                <span style={{ fontSize: 15, fontWeight: 800, color: '#0f172a' }}>{label}</span>
                <span style={{ fontSize: 12, color: '#94a3b8', fontWeight: 500 }}>{sub}</span>
                <div style={{ flex: 1, height: 1, background: '#e2e8f0', marginLeft: 8 }} />
                <span style={{
                  fontSize: 10, fontWeight: 700, color: '#4f46e5',
                  background: '#ede9fe', padding: '2px 8px', borderRadius: 99,
                }}>{games.length}G</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {games.map(g => <GameRow key={g.game_id} g={g} />)}
              </div>
            </section>
          )
        })}
      </div>
    </div>
  )
}
