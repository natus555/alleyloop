import { useNavigate } from 'react-router-dom'
import { useApi } from '../hooks/useApi'
import TeamBadge from '../components/TeamBadge'

interface UpcomingGame {
  game_id: string
  home_tri: string; away_tri: string
  home_name: string; away_name: string
  home_color: string; away_color: string
  home_logo?: string; away_logo?: string
  win_prob_home?: number
  win_prob_away?: number
  home_score?: number
  away_score?: number
}

interface DayGroup { date: string; games: UpcomingGame[] }

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
  const hasPred = typeof g.win_prob_home === 'number' && typeof g.win_prob_away === 'number' && typeof g.home_score === 'number' && typeof g.away_score === 'number'

  const homeWin = hasPred ? g.win_prob_home! >= g.win_prob_away! : null
  const favColor = homeWin === null ? '#4f46e5' : homeWin ? g.home_color : g.away_color
  const favPct = hasPred ? Math.round((homeWin ? g.win_prob_home! : g.win_prob_away!) * 100) : null

  return (
    <div onClick={() => nav(`/game/${id}`)} style={{
      background: '#ffffff',
      border: '1px solid #e2e8f0',
      borderRadius: 12, padding: '16px 20px',
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
        {hasPred ? (
          <>
            <div style={{ fontSize: 28, fontWeight: 900, letterSpacing: '-1.5px', color: '#0f172a', lineHeight: 1 }}>
              <span style={{ color: g.home_color }}>{Math.round(g.home_score!)}</span>
              <span style={{ color: '#cbd5e1', margin: '0 8px' }}>–</span>
              <span style={{ color: g.away_color }}>{Math.round(g.away_score!)}</span>
            </div>
            <div style={{ fontSize: 10, color: '#94a3b8', margin: '4px 0 6px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>projected</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
              <div style={{ width: 76, height: 4, borderRadius: 3, overflow: 'hidden', background: '#e2e8f0', display: 'flex' }}>
                <div style={{ height: '100%', width: `${Math.round(g.win_prob_home! * 100)}%`, background: g.home_color }} />
                <div style={{ height: '100%', width: `${Math.round(g.win_prob_away! * 100)}%`, background: g.away_color }} />
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

export default function Upcoming() {
  const { data, loading } = useApi<{ days: DayGroup[] }>('/schedule/upcoming')

  const fmt = (d: string) => {
    const dt = new Date(d + 'T12:00:00')
    const today = new Date()
    const diff = Math.round((dt.getTime() - today.getTime()) / 86400000)
    const label = diff === 0 ? 'Today' : diff === 1 ? 'Tomorrow' : dt.toLocaleDateString('en-US', { weekday: 'long' })
    return { label, sub: dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) }
  }

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

      {!loading && !data?.days.length && (
        <div style={{ textAlign: 'center', padding: '60px 0', color: '#94a3b8', fontSize: 14 }}>
          No upcoming games in the next 7 days
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
        {data?.days.map(day => {
          const { label, sub } = fmt(day.date)
          return (
            <section key={day.date}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 10 }}>
                <span style={{ fontSize: 15, fontWeight: 800, color: '#0f172a' }}>{label}</span>
                <span style={{ fontSize: 12, color: '#94a3b8', fontWeight: 500 }}>{sub}</span>
                <div style={{ flex: 1, height: 1, background: '#e2e8f0', marginLeft: 8 }} />
                <span style={{
                  fontSize: 10, fontWeight: 700, color: '#4f46e5',
                  background: '#ede9fe', padding: '2px 8px', borderRadius: 99,
                }}>{day.games.length}G</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {day.games.map(g => <GameRow key={`${g.home_tri}-${g.away_tri}`} g={g} />)}
              </div>
            </section>
          )
        })}
      </div>
    </div>
  )
}
