import { useState } from 'react'
import { useApi } from '../hooks/useApi'
import StatusBadge from '../components/StatusBadge'
import TeamBadge from '../components/TeamBadge'

interface InjuryRow {
  team_tri: string; player_name: string
  position: string; status: string
  status_rank: number; injury_detail: string
}

interface Team { tri: string; name: string; color: string }

const STATUS_ORDER = ['Out', 'Injured Reserve', 'Suspension', 'Doubtful', 'Questionable', 'Game Time Decision', 'Day-To-Day', 'Probable']
function rankStatus(s: string) {
  const i = STATUS_ORDER.findIndex(x => s.toLowerCase().includes(x.toLowerCase()))
  return i === -1 ? 99 : i
}

export default function Injuries() {
  const [team, setTeam] = useState('')
  const [search, setSearch] = useState('')
  const { data: teams } = useApi<Team[]>('/teams')
  const { data, loading } = useApi<{ injuries: InjuryRow[] }>(
    `/injuries${team ? `?team=${team}` : ''}`, [team], 600_000
  )

  const rows = (data?.injuries ?? [])
    .filter(r => !search || r.player_name.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => rankStatus(a.status) - rankStatus(b.status))

  const grouped: Record<string, InjuryRow[]> = {}
  rows.forEach(r => {
    if (!grouped[r.team_tri]) grouped[r.team_tri] = []
    grouped[r.team_tri].push(r)
  })

  const outCount   = rows.filter(r => r.status_rank >= 4).length
  const questCount = rows.filter(r => r.status_rank === 2 || r.status_rank === 3).length

  const inputStyle = {
    background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8,
    padding: '9px 14px', fontSize: 13, color: '#0f172a', outline: 'none',
  }

  return (
    <div className="fade-in">
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 26, fontWeight: 900, color: '#0f172a', letterSpacing: '-0.5px', marginBottom: 4 }}>
          Injury <span className="gradient-text">Report</span>
        </h1>
        <p style={{ fontSize: 13, color: '#94a3b8', fontWeight: 500 }}>Real-time from ESPN · updates every 10 min</p>
      </div>

      {/* Summary pills */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap' }}>
        {[
          { label: 'Out / IR', count: outCount, bg: '#fee2e2', text: '#dc2626' },
          { label: 'Questionable', count: questCount, bg: '#fef3c7', text: '#d97706' },
          { label: 'Total Listed', count: rows.length, bg: '#ede9fe', text: '#6d28d9' },
        ].map(p => (
          <div key={p.label} style={{
            padding: '8px 16px', display: 'flex', gap: 8, alignItems: 'center',
            background: p.bg, borderRadius: 8, border: `1px solid ${p.bg}`,
          }}>
            <span style={{ fontSize: 22, fontWeight: 800, color: p.text, lineHeight: 1 }}>{p.count}</span>
            <span style={{ fontSize: 12, color: p.text, fontWeight: 600 }}>{p.label}</span>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap' }}>
        <input
          placeholder="Search player…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{ ...inputStyle, width: 220 }}
        />
        <select value={team} onChange={e => setTeam(e.target.value)} style={{ ...inputStyle, color: team ? '#0f172a' : '#94a3b8' }}>
          <option value="">All Teams</option>
          {teams?.map(t => <option key={t.tri} value={t.tri}>{t.name}</option>)}
        </select>
      </div>

      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {[...Array(8)].map((_, i) => <div key={i} className="shimmer" style={{ height: 44, borderRadius: 8 }} />)}
        </div>
      )}

      {!loading && rows.length === 0 && (
        <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 12, padding: 60, textAlign: 'center', color: '#94a3b8', boxShadow: '0 1px 4px rgba(0,0,0,0.04)' }}>
          No injuries reported
        </div>
      )}

      {/* Single-team table */}
      {!loading && team ? (
        <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 12, overflow: 'hidden', boxShadow: '0 1px 4px rgba(0,0,0,0.05)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                {['Player', 'Pos', 'Status', 'Detail'].map(h => (
                  <th key={h} style={{ padding: '10px 16px', textAlign: 'left', fontSize: 11, color: '#94a3b8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f8fafc', transition: 'background 0.1s' }}
                  onMouseEnter={e => (e.currentTarget.style.background = '#fafbff')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                  <td style={{ padding: '10px 16px', fontWeight: 600, fontSize: 13, color: '#0f172a' }}>{r.player_name}</td>
                  <td style={{ padding: '10px 16px', fontSize: 12, color: '#64748b' }}>{r.position}</td>
                  <td style={{ padding: '10px 16px' }}><StatusBadge status={r.status} /></td>
                  <td style={{ padding: '10px 16px', fontSize: 12, color: '#94a3b8', maxWidth: 300 }}>{r.injury_detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        /* All-teams grouped view */
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {Object.entries(grouped).sort(([a], [b]) => a.localeCompare(b)).map(([tri, players]) => {
            const teamInfo = teams?.find(t => t.tri === tri)
            return (
              <div key={tri} style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 12, overflow: 'hidden', boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
                <div style={{
                  padding: '10px 16px', borderBottom: '1px solid #f1f5f9',
                  display: 'flex', alignItems: 'center', gap: 10, background: '#f8fafc',
                }}>
                  <TeamBadge tri={tri} color={teamInfo?.color} size={28} />
                  <span style={{ fontWeight: 700, fontSize: 13, color: '#0f172a' }}>{teamInfo?.name ?? tri}</span>
                  <span style={{ fontSize: 11, color: '#94a3b8', marginLeft: 4 }}>{players.length} listed</span>
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <tbody>
                    {players.map((r, i) => (
                      <tr key={i} style={{ borderBottom: i < players.length - 1 ? '1px solid #f8fafc' : 'none' }}
                        onMouseEnter={e => (e.currentTarget.style.background = '#fafbff')}
                        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                        <td style={{ padding: '9px 16px', fontWeight: 600, fontSize: 13, color: '#0f172a', width: '30%' }}>{r.player_name}</td>
                        <td style={{ padding: '9px 16px', fontSize: 11, color: '#94a3b8', width: 40, fontWeight: 600 }}>{r.position}</td>
                        <td style={{ padding: '9px 16px', width: 160 }}><StatusBadge status={r.status} small /></td>
                        <td style={{ padding: '9px 16px', fontSize: 12, color: '#94a3b8' }}>{r.injury_detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
