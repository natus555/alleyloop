import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { Activity, Calendar, AlertTriangle, Users, TrendingUp, Shuffle, ListOrdered } from 'lucide-react'
import Live from './pages/Live'
import Upcoming from './pages/Upcoming'
import Injuries from './pages/Injuries'
import LineupOptimizer from './pages/LineupOptimizer'
import GameDetail from './pages/GameDetail'
import Rosters from './pages/Rosters'

const NAV = [
  { to: '/',          icon: Activity,      label: 'Live'      },
  { to: '/upcoming',  icon: Calendar,      label: 'Schedule'  },
  { to: '/injuries',  icon: AlertTriangle, label: 'Injuries'  },
  { to: '/rosters',   icon: ListOrdered,   label: 'Rosters'   },
  { to: '/optimizer', icon: Users,         label: 'Optimizer' },
]

function NavBar() {
  const loc = useLocation()
  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 50,
      background: '#0f172a',
      borderBottom: '3px solid #4f46e5',
      boxShadow: '0 2px 12px rgba(0,0,0,0.25)',
    }}>
      <div style={{
        maxWidth: 1200, margin: '0 auto',
        display: 'flex', alignItems: 'center',
        height: 56, padding: '0 20px', gap: 4,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginRight: 24, flexShrink: 0 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 7,
            background: 'linear-gradient(135deg, #4f46e5, #7c3aed)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <TrendingUp size={15} color="#fff" />
          </div>
          <span style={{ fontSize: 17, fontWeight: 800, color: '#ffffff', letterSpacing: '-0.4px' }}>
            AlleyLoop
          </span>
        </div>

        {NAV.map(({ to, icon: Icon, label }) => {
          const active = to === '/' ? loc.pathname === '/' : loc.pathname.startsWith(to)
          return (
            <NavLink
              key={to}
              to={to}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '6px 14px',
                borderRadius: 8,
                fontSize: 13, fontWeight: 600,
                textDecoration: 'none',
                color: active ? '#ffffff' : '#94a3b8',
                background: active ? 'rgba(255,255,255,0.1)' : 'transparent',
                borderBottom: active ? '2px solid #818cf8' : '2px solid transparent',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={e => {
                if (!active) {
                  (e.currentTarget as HTMLAnchorElement).style.color = '#e2e8f0'
                  ;(e.currentTarget as HTMLAnchorElement).style.background = 'rgba(255,255,255,0.06)'
                }
              }}
              onMouseLeave={e => {
                if (!active) {
                  (e.currentTarget as HTMLAnchorElement).style.color = '#94a3b8'
                  ;(e.currentTarget as HTMLAnchorElement).style.background = 'transparent'
                }
              }}
            >
              <Icon size={14} />
              {label}
            </NavLink>
          )
        })}

        <div style={{
          marginLeft: 'auto', display: 'flex', alignItems: 'center',
          gap: 6, opacity: 0.5,
        }}>
          <Shuffle size={13} color="#94a3b8" />
          <span style={{ fontSize: 12, color: '#94a3b8' }}>Trade Advisor</span>
          <span style={{
            fontSize: 10, color: '#64748b', fontWeight: 700, letterSpacing: '0.05em',
            background: 'rgba(255,255,255,0.08)', padding: '2px 7px', borderRadius: 99,
          }}>SOON</span>
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <main style={{ maxWidth: 1200, margin: '0 auto', padding: '76px 20px 48px' }}>
        <Routes>
          <Route path="/"             element={<Live />} />
          <Route path="/upcoming"     element={<Upcoming />} />
          <Route path="/injuries"     element={<Injuries />} />
          <Route path="/rosters"      element={<Rosters />} />
          <Route path="/optimizer"    element={<LineupOptimizer />} />
          <Route path="/game/:gameId" element={<GameDetail />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
