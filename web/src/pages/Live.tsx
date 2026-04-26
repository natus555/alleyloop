import { useNavigate } from 'react-router-dom'
import { useApi } from '../hooks/useApi'
import TeamBadge from '../components/TeamBadge'

interface LiveGame {
  homeTeam: string; awayTeam: string
  home_name: string; away_name: string
  home_color: string; away_color: string
  home_logo?: string; away_logo?: string
  homeScore: number; awayScore: number
  period: number; gameClock: string
  status: 'live' | 'final' | 'scheduled'
}

interface TodayGame {
  game_id: string
  home_tri: string; away_tri: string
  home_name: string; away_name: string
  home_color: string; away_color: string
  home_logo?: string; away_logo?: string
  status: string; start_time: string
  home_score: number | null; away_score: number | null
  venue: string
}

interface UpcomingGame {
  game_id: string
  home_tri: string; away_tri: string
  home_name: string; away_name: string
  home_color: string; away_color: string
  home_logo?: string; away_logo?: string
}

interface DayGroup { date: string; games: UpcomingGame[] }

interface Pred {
  win_prob_home: number; win_prob_away: number
  home_score: number; away_score: number
  home_color: string; away_color: string
}

type UnifiedGame = {
  id: string
  home_tri: string; away_tri: string
  home_name: string; away_name: string
  home_color: string; away_color: string
  home_logo?: string; away_logo?: string
  home_score: number | null; away_score: number | null
  status: 'live' | 'final' | 'upcoming'
  period?: number; clock?: string
  start_time?: string; venue?: string
  sortKey: number
}

function PredBadge({ home, away, hc, ac }: { home: string; away: string; hc: string; ac: string }) {
  const { data } = useApi<Pred>(`/predict/game?home=${home}&away=${away}`)
  if (!data) return (
    <div style={{ textAlign: 'center' }}>
      <div className="shimmer" style={{ height: 28, width: 130, margin: '0 auto', borderRadius: 6 }} />
    </div>
  )
  const homeWin = data.win_prob_home >= data.win_prob_away
  const favTri = homeWin ? home : away
  const favPct = Math.round((homeWin ? data.win_prob_home : data.win_prob_away) * 100)
  const favColor = homeWin ? hc : ac
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 24, fontWeight: 900, letterSpacing: '-1.5px', color: '#0f172a', marginBottom: 2 }}>
        <span style={{ color: hc }}>{Math.round(data.home_score)}</span>
        <span style={{ color: '#cbd5e1', margin: '0 6px' }}>–</span>
        <span style={{ color: ac }}>{Math.round(data.away_score)}</span>
      </div>
      <div style={{ fontSize: 10, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 3 }}>projected</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
        <div style={{ width: 72, height: 3, borderRadius: 2, overflow: 'hidden', background: '#e2e8f0' }}>
          <div style={{
            height: '100%', width: `${Math.round(data.win_prob_home * 100)}%`,
            background: `linear-gradient(90deg, ${hc}, ${ac})`,
          }} />
        </div>
        <span style={{ fontSize: 11, fontWeight: 700, color: favColor }}>{favTri} {favPct}%</span>
      </div>
    </div>
  )
}

function TeamCol({ tri, name, color, logo, side }: { tri: string; name: string; color: string; logo?: string; side: 'home' | 'away' }) {
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

function GameCard({ g }: { g: UnifiedGame }) {
  const nav = useNavigate()
  const isLive  = g.status === 'live'
  const isFinal = g.status === 'final'
  const hasScore = g.home_score != null && g.away_score != null

  const timeStr = g.start_time
    ? new Date(g.start_time).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZoneName: 'short' })
    : ''

  return (
    <div onClick={() => nav(`/game/${g.id}`)} style={{
      background: '#ffffff',
      border: `1px solid ${isLive ? 'rgba(239,68,68,0.3)' : '#e2e8f0'}`,
      borderRadius: 12,
      padding: '16px 20px',
      cursor: 'pointer',
      transition: 'all 0.18s ease',
      position: 'relative',
      overflow: 'hidden',
      boxShadow: '0 1px 4px rgba(0,0,0,0.05)',
    }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLDivElement).style.boxShadow = '0 6px 20px rgba(0,0,0,0.1)'
        ;(e.currentTarget as HTMLDivElement).style.transform = 'translateY(-2px)'
        ;(e.currentTarget as HTMLDivElement).style.borderColor = isLive ? 'rgba(239,68,68,0.5)' : '#a5b4fc'
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLDivElement).style.boxShadow = '0 1px 4px rgba(0,0,0,0.05)'
        ;(e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)'
        ;(e.currentTarget as HTMLDivElement).style.borderColor = isLive ? 'rgba(239,68,68,0.3)' : '#e2e8f0'
      }}>

      {/* Live stripe */}
      {isLive && <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 3, background: 'linear-gradient(90deg,#ef4444,#f97316)' }} />}

      {/* Status row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {isLive && <span className="live-dot" />}
          <span style={{
            fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase',
            color: isLive ? '#ef4444' : isFinal ? '#94a3b8' : '#4f46e5',
          }}>
            {isLive ? `Q${g.period} · ${g.clock}` : isFinal ? 'Final' : timeStr || 'Upcoming'}
          </span>
        </div>
        {g.venue && <span style={{ fontSize: 10, color: '#cbd5e1' }}>{g.venue}</span>}
      </div>

      {/* Teams + score */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', gap: 12 }}>
        <TeamCol tri={g.home_tri} name={g.home_name} color={g.home_color} logo={g.home_logo} side="home" />

        <div style={{ textAlign: 'center', minWidth: 150 }}>
          {hasScore ? (
            <div>
              <div style={{ fontSize: 34, fontWeight: 900, letterSpacing: '-2px', lineHeight: 1 }}>
                <span style={{ color: g.home_color }}>{g.home_score}</span>
                <span style={{ color: '#e2e8f0', margin: '0 8px' }}>:</span>
                <span style={{ color: g.away_color }}>{g.away_score}</span>
              </div>
              {isFinal && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Final</div>}
            </div>
          ) : (
            <PredBadge home={g.home_tri} away={g.away_tri} hc={g.home_color} ac={g.away_color} />
          )}
        </div>

        <TeamCol tri={g.away_tri} name={g.away_name} color={g.away_color} logo={g.away_logo} side="away" />
      </div>
    </div>
  )
}

function SectionLabel({ label, count }: { label: string; count: number }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, margin: '28px 0 10px' }}>
      <span style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</span>
      <div style={{ flex: 1, height: 1, background: '#e2e8f0' }} />
      <span style={{
        fontSize: 10, fontWeight: 700, color: '#4f46e5',
        background: '#ede9fe', padding: '2px 8px', borderRadius: 99,
      }}>{count}</span>
    </div>
  )
}

export default function Live() {
  const { data: liveData } = useApi<{ games: LiveGame[] }>('/live', [], 5000)
  const { data: todayData } = useApi<{ games: TodayGame[] }>('/schedule/today', [], 30000)
  const { data: upcomingData } = useApi<{ days: DayGroup[] }>('/schedule/upcoming', [], 60000)

  const unified: UnifiedGame[] = []
  const seenIds = new Set<string>()

  ;(liveData?.games ?? []).forEach(g => {
    const id = `${g.homeTeam}-vs-${g.awayTeam}`
    if (seenIds.has(id)) return
    seenIds.add(id)
    const status: UnifiedGame['status'] = g.status === 'scheduled' ? 'upcoming' : g.status
    unified.push({
      id,
      home_tri: g.homeTeam, away_tri: g.awayTeam,
      home_name: g.home_name, away_name: g.away_name,
      home_color: g.home_color, away_color: g.away_color,
      home_logo: g.home_logo, away_logo: g.away_logo,
      home_score: g.status !== 'scheduled' ? g.homeScore : null,
      away_score: g.status !== 'scheduled' ? g.awayScore : null,
      status,
      period: g.period, clock: g.gameClock,
      sortKey: g.status === 'live' ? 0 : g.status === 'final' ? 2 : 1,
    })
  })

  ;(todayData?.games ?? []).forEach(g => {
    const id = `${g.home_tri}-vs-${g.away_tri}`
    if (seenIds.has(id)) return
    seenIds.add(id)
    const statusLow = g.status.toLowerCase()
    const isFinal = statusLow.includes('final') || g.status === '3'
    const isLive  = statusLow.includes('progress') || g.status === '2'
    unified.push({
      id,
      home_tri: g.home_tri, away_tri: g.away_tri,
      home_name: g.home_name, away_name: g.away_name,
      home_color: g.home_color, away_color: g.away_color,
      home_logo: g.home_logo, away_logo: g.away_logo,
      home_score: g.home_score, away_score: g.away_score,
      status: isLive ? 'live' : isFinal ? 'final' : 'upcoming',
      start_time: g.start_time, venue: g.venue,
      sortKey: isLive ? 0 : isFinal ? 2 : 1,
    })
  })

  const tomorrow = upcomingData?.days?.[0]
  if (tomorrow) {
    ;(tomorrow.games ?? []).forEach(g => {
      const id = `${g.home_tri}-vs-${g.away_tri}`
      if (seenIds.has(id)) return
      seenIds.add(id)
      unified.push({
        id,
        home_tri: g.home_tri, away_tri: g.away_tri,
        home_name: g.home_name, away_name: g.away_name,
        home_color: g.home_color, away_color: g.away_color,
        home_logo: g.home_logo, away_logo: g.away_logo,
        home_score: null, away_score: null,
        status: 'upcoming', sortKey: 3,
      })
    })
  }

  unified.sort((a, b) => a.sortKey - b.sortKey)
  const liveGames     = unified.filter(g => g.status === 'live')
  const upcomingGames = unified.filter(g => g.status === 'upcoming')
  const finalGames    = unified.filter(g => g.status === 'final')
  const totalLoading = !liveData && !todayData

  return (
    <div className="fade-in" style={{ maxWidth: 900, margin: '0 auto' }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 28, fontWeight: 900, color: '#0f172a', letterSpacing: '-0.8px', marginBottom: 4 }}>
          Live <span className="gradient-text">Games</span>
        </h1>
        <p style={{ fontSize: 12, color: '#94a3b8', fontWeight: 500 }}>
          Live scores · predictions for upcoming games · click any game for full analysis
        </p>
      </div>

      {totalLoading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[1, 2, 3].map(i => <div key={i} className="shimmer" style={{ height: 100, borderRadius: 12 }} />)}
        </div>
      )}

      {liveGames.length > 0 && (
        <>
          <SectionLabel label="In Progress" count={liveGames.length} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {liveGames.map(g => <GameCard key={g.id} g={g} />)}
          </div>
        </>
      )}

      {upcomingGames.length > 0 && (
        <>
          <SectionLabel label="Upcoming Today" count={upcomingGames.length} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {upcomingGames.map(g => <GameCard key={g.id} g={g} />)}
          </div>
        </>
      )}

      {finalGames.length > 0 && (
        <>
          <SectionLabel label="Recent Results" count={finalGames.length} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {finalGames.map(g => <GameCard key={g.id} g={g} />)}
          </div>
        </>
      )}

      {!totalLoading && unified.length === 0 && (
        <div style={{ textAlign: 'center', padding: '60px 0', color: '#94a3b8', fontSize: 14 }}>
          No games in the last or next 24 hours
        </div>
      )}
    </div>
  )
}
