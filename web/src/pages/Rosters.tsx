import { useState } from 'react'
import { useApi } from '../hooks/useApi'
import TeamBadge from '../components/TeamBadge'

interface Team {
  tri: string
  name: string
  color: string
}

interface RosterPlayer {
  player_name: string
  position: string
  source: string
}

interface TeamRoster {
  team_tri: string
  team_name: string
  team_color: string
  team_logo?: string
  players: RosterPlayer[]
}

interface TradeRow {
  player_name: string
  to_team_tri: string
  transaction_type: string
  description: string
  transaction_date: string
}

export default function Rosters() {
  const [team, setTeam] = useState('')
  const [search, setSearch] = useState('')
  const { data: teams } = useApi<Team[]>('/teams')
  const { data, loading } = useApi<{ rosters: TeamRoster[] }>(`/rosters${team ? `?team=${team}` : ''}`, [team], 300000)
  const { data: trades } = useApi<{ trades: TradeRow[] }>('/trades/recent', [], 300000)

  const rosters = (data?.rosters ?? []).map(r => ({
    ...r,
    players: r.players.filter(p => !search || p.player_name.toLowerCase().includes(search.toLowerCase())),
  })).filter(r => r.players.length > 0)

  const totalPlayers = rosters.reduce((n, r) => n + r.players.length, 0)
  const tradeAdjusted = rosters.reduce((n, r) => n + r.players.filter(p => p.source === 'trade_adjusted').length, 0)

  return (
    <div className="fade-in">
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 26, fontWeight: 900, color: '#0f172a', letterSpacing: '-0.5px', marginBottom: 4 }}>
          Live <span className="gradient-text">Rosters</span>
        </h1>
        <p style={{ fontSize: 13, color: '#94a3b8', fontWeight: 500 }}>
          ESPN rosters + recent trade adjustments
        </p>
      </div>

      <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap' }}>
        <input
          placeholder="Search player..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '9px 14px', fontSize: 13, width: 220 }}
        />
        <select
          value={team}
          onChange={e => setTeam(e.target.value)}
          style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '9px 14px', fontSize: 13 }}
        >
          <option value="">All Teams</option>
          {teams?.map(t => <option key={t.tri} value={t.tri}>{t.name}</option>)}
        </select>
      </div>

      <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap' }}>
        <div style={{ padding: '8px 16px', background: '#ede9fe', borderRadius: 8 }}>
          <span style={{ fontSize: 22, fontWeight: 800, color: '#6d28d9', marginRight: 8 }}>{totalPlayers}</span>
          <span style={{ fontSize: 12, color: '#6d28d9', fontWeight: 600 }}>Visible players</span>
        </div>
        <div style={{ padding: '8px 16px', background: '#dcfce7', borderRadius: 8 }}>
          <span style={{ fontSize: 22, fontWeight: 800, color: '#15803d', marginRight: 8 }}>{tradeAdjusted}</span>
          <span style={{ fontSize: 12, color: '#15803d', fontWeight: 600 }}>Trade-adjusted</span>
        </div>
      </div>

      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[...Array(5)].map((_, i) => <div key={i} className="shimmer" style={{ height: 72, borderRadius: 10 }} />)}
        </div>
      )}

      {!loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {rosters.map(r => (
            <div key={r.team_tri} style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 12, overflow: 'hidden' }}>
              <div style={{ padding: '10px 14px', background: '#f8fafc', display: 'flex', alignItems: 'center', gap: 10, borderBottom: '1px solid #f1f5f9' }}>
                <TeamBadge tri={r.team_tri} color={r.team_color} size={28} logo={r.team_logo} />
                <span style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>{r.team_name}</span>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>{r.players.length} players</span>
              </div>
              <div style={{ padding: 12, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 8 }}>
                {r.players.map((p, i) => (
                  <div key={`${p.player_name}-${i}`} style={{ border: '1px solid #f1f5f9', borderRadius: 8, padding: '8px 10px', display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                    <span style={{ fontSize: 12, color: '#0f172a', fontWeight: 600 }}>{p.player_name}</span>
                    <span style={{ fontSize: 10, color: '#64748b', fontWeight: 700 }}>{p.position}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <div style={{ marginTop: 20, background: '#fff', border: '1px solid #e2e8f0', borderRadius: 12, overflow: 'hidden' }}>
        <div style={{ padding: '10px 14px', background: '#f8fafc', borderBottom: '1px solid #f1f5f9', fontSize: 12, fontWeight: 700, color: '#475569' }}>
          Recent Trades Applied
        </div>
        <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {(trades?.trades ?? []).slice(0, 12).map((t, i) => (
            <div key={`${t.player_name}-${i}`} style={{ fontSize: 12, color: '#334155' }}>
              <strong>{t.player_name}</strong> {'->'} {t.to_team_tri} {t.transaction_date ? `(${t.transaction_date})` : ''}
            </div>
          ))}
          {(trades?.trades ?? []).length === 0 && <div style={{ fontSize: 12, color: '#94a3b8' }}>No recent trades found.</div>}
        </div>
      </div>
    </div>
  )
}
