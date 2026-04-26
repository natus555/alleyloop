import { useState } from 'react'
import { useApi } from '../hooks/useApi'
import StatusBadge from '../components/StatusBadge'
import TeamBadge from '../components/TeamBadge'

interface InjuryRow {
  team_tri: string; player_name: string
  position: string; status: string
  status_rank: number; injury_detail: string
}

interface RiskPlayer {
  player_name: string; team_tri: string; team_name: string; team_color: string
  risk_pct: number; risk_level: 'High' | 'Medium' | 'Low'
  min_avg: number; min_recent: number; rest_days: number; is_b2b: boolean
  inj_status: string; inj_detail: string; drivers: string[]
}

interface Team { tri: string; name: string; color: string }

const STATUS_ORDER = ['Out', 'Injured Reserve', 'Suspension', 'Doubtful', 'Questionable', 'Game Time Decision', 'Day-To-Day', 'Probable']
function rankStatus(s: string) {
  const i = STATUS_ORDER.findIndex(x => s.toLowerCase().includes(x.toLowerCase()))
  return i === -1 ? 99 : i
}

const RISK_COLORS = {
  High:   { bg: '#fee2e2', border: '#fca5a5', text: '#dc2626', bar: '#ef4444' },
  Medium: { bg: '#fef3c7', border: '#fde68a', text: '#d97706', bar: '#f59e0b' },
  Low:    { bg: '#dcfce7', border: '#86efac', text: '#15803d', bar: '#22c55e' },
}

function RiskBar({ pct, level }: { pct: number; level: 'High' | 'Medium' | 'Low' }) {
  const c = RISK_COLORS[level]
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 5, background: '#f1f5f9', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: c.bar, borderRadius: 3, transition: 'width 0.6s ease' }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 800, color: c.text, width: 36, textAlign: 'right' }}>{pct}%</span>
    </div>
  )
}

function RiskCard({ p }: { p: RiskPlayer }) {
  const c = RISK_COLORS[p.risk_level]
  return (
    <div style={{
      background: '#fff', border: `1px solid ${c.border}`,
      borderRadius: 12, padding: '14px 16px',
      boxShadow: '0 1px 4px rgba(0,0,0,0.04)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <TeamBadge tri={p.team_tri} color={p.team_color} size={28} />
          <div>
            <div style={{ fontWeight: 700, fontSize: 13, color: '#0f172a' }}>{p.player_name}</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 1 }}>{p.team_name}</div>
          </div>
        </div>
        <span style={{
          fontSize: 10, fontWeight: 800, padding: '3px 8px', borderRadius: 99,
          background: c.bg, color: c.text, border: `1px solid ${c.border}`,
          letterSpacing: '0.04em',
        }}>{p.risk_level.toUpperCase()} RISK</span>
      </div>

      <RiskBar pct={p.risk_pct} level={p.risk_level} />

      <div style={{ display: 'flex', gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
        {p.drivers.map((d, i) => (
          <span key={i} style={{
            fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 6,
            background: '#f1f5f9', color: '#64748b',
          }}>{d}</span>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 16, marginTop: 10, fontSize: 11, color: '#94a3b8' }}>
        <span>Avg <strong style={{ color: '#0f172a' }}>{p.min_avg}</strong> min</span>
        <span>Recent <strong style={{ color: p.min_recent > p.min_avg * 1.1 ? '#ef4444' : '#0f172a' }}>{p.min_recent}</strong> min</span>
        <span>Rest <strong style={{ color: p.rest_days <= 1 ? '#ef4444' : '#0f172a' }}>{p.rest_days}d</strong></span>
        {p.is_b2b && <span style={{ color: '#f59e0b', fontWeight: 700 }}>B2B</span>}
        {p.inj_status && <StatusBadge status={p.inj_status} small />}
      </div>
    </div>
  )
}

export default function Injuries() {
  const [team, setTeam]     = useState('')
  const [search, setSearch] = useState('')
  const [riskTeam, setRiskTeam] = useState('')

  const { data: teams }  = useApi<Team[]>('/teams')
  const { data, loading } = useApi<{ injuries: InjuryRow[] }>(
    `/injuries${team ? `?team=${team}` : ''}`, [team], 600_000
  )
  const { data: riskData, loading: riskLoading } = useApi<{ players: RiskPlayer[] }>(
    `/injuries/risk${riskTeam ? `?team=${riskTeam}` : ''}`, [riskTeam], 300_000
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

  const riskPlayers = riskData?.players ?? []
  const highRisk   = riskPlayers.filter(p => p.risk_level === 'High')
  const medRisk    = riskPlayers.filter(p => p.risk_level === 'Medium')

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
        <p style={{ fontSize: 13, color: '#94a3b8', fontWeight: 500 }}>Real-time from ESPN · load-based risk predictions from player tracking data</p>
      </div>

      <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 16, padding: 20, marginBottom: 28, boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <div style={{ fontWeight: 800, fontSize: 15, color: '#0f172a' }}>Injury Risk Watch List</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>Based on minutes load, rest days, usage spike &amp; injury history</div>
          </div>
          <select value={riskTeam} onChange={e => setRiskTeam(e.target.value)} style={{ ...inputStyle, fontSize: 12, padding: '6px 12px', color: riskTeam ? '#0f172a' : '#94a3b8' }}>
            <option value="">All Teams</option>
            {teams?.map(t => <option key={t.tri} value={t.tri}>{t.tri}</option>)}
          </select>
        </div>

        {riskLoading && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: 10 }}>
            {[...Array(6)].map((_, i) => <div key={i} className="shimmer" style={{ height: 120, borderRadius: 12 }} />)}
          </div>
        )}

        {!riskLoading && highRisk.length > 0 && (
          <>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#dc2626', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
              High Risk · {highRisk.length} players
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: 10, marginBottom: 16 }}>
              {highRisk.map(p => <RiskCard key={`${p.team_tri}-${p.player_name}`} p={p} />)}
            </div>
          </>
        )}

        {!riskLoading && medRisk.length > 0 && (
          <>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#d97706', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
              Medium Risk · {medRisk.length} players
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: 10 }}>
              {medRisk.slice(0, riskTeam ? undefined : 12).map(p => <RiskCard key={`${p.team_tri}-${p.player_name}`} p={p} />)}
              {!riskTeam && medRisk.length > 12 && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', fontSize: 12, fontWeight: 600 }}>
                  +{medRisk.length - 12} more · filter by team to see all
                </div>
              )}
            </div>
          </>
        )}

        {!riskLoading && highRisk.length === 0 && medRisk.length === 0 && (
          <div style={{ textAlign: 'center', padding: '24px 0', color: '#94a3b8', fontSize: 13 }}>No elevated risk players detected</div>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 16 }}>
        <span style={{ fontSize: 15, fontWeight: 800, color: '#0f172a' }}>ESPN Injury Report</span>
        <div style={{ flex: 1, height: 1, background: '#e2e8f0' }} />
      </div>

      <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap' }}>
        {[
          { label: 'Out / IR', count: outCount, bg: '#fee2e2', text: '#dc2626' },
          { label: 'Questionable', count: questCount, bg: '#fef3c7', text: '#d97706' },
          { label: 'Total Listed', count: rows.length, bg: '#ede9fe', text: '#6d28d9' },
        ].map(p => (
          <div key={p.label} style={{
            padding: '7px 14px', display: 'flex', gap: 8, alignItems: 'center',
            background: p.bg, borderRadius: 8,
          }}>
            <span style={{ fontSize: 20, fontWeight: 800, color: p.text, lineHeight: 1 }}>{p.count}</span>
            <span style={{ fontSize: 12, color: p.text, fontWeight: 600 }}>{p.label}</span>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap' }}>
        <input placeholder="Search player…" value={search} onChange={e => setSearch(e.target.value)} style={{ ...inputStyle, width: 220 }} />
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
        <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 12, padding: 60, textAlign: 'center', color: '#94a3b8' }}>
          No injuries reported
        </div>
      )}

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
                <tr key={i} style={{ borderBottom: '1px solid #f8fafc' }}
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
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {Object.entries(grouped).sort(([a], [b]) => a.localeCompare(b)).map(([tri, players]) => {
            const teamInfo = teams?.find(t => t.tri === tri)
            return (
              <div key={tri} style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 12, overflow: 'hidden', boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
                <div style={{ padding: '10px 16px', borderBottom: '1px solid #f1f5f9', display: 'flex', alignItems: 'center', gap: 10, background: '#f8fafc' }}>
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
