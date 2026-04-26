import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { Activity, Calendar, AlertTriangle, Users, TrendingUp, Shuffle } from 'lucide-react'
import Live from './pages/Live'
import Upcoming from './pages/Upcoming'
import Injuries from './pages/Injuries'
import LineupOptimizer from './pages/LineupOptimizer'
import GameDetail from './pages/GameDetail'

const NAV = [
  { to: '/',          icon: Activity,       label: 'Live'      },
  { to: '/upcoming',  icon: Calendar,       label: 'Schedule'  },
  { to: '/injuries',  icon: AlertTriangle,  label: 'Injuries'  },
  { to: '/optimizer', icon: Users,          label: 'Optimizer' },
]

function NavBar() {
  const loc = useLocation()
  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 50,
      background: '#0f172a',
      borderBottom: '3px solid #4f46e5',
      boxShadow: '0 2px 12px rgba(0,0,0,0.15)',
    }}>
      <div style={{ maxWidth: 1280, margin: '0 auto', padding: '0 20px', display: 'flex', alignItems: 'center', height: 54, gap: 2 }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginRight: 28, flexShrink: 0 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 6, background: 'linear-gradient(135deg, #4f46e5, #7c3aed)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <TrendingUp size={15} color="#fff" />
          </div>
          <span style={{ fontWeight: 800, color: '#fff', letterSpacing: '-0.5px', fontSize: 17 }}>AlleyLoop</span>
        </div>

        {/* Nav links */}
        {NAV.map(({ to, icon: Icon, label }) => {
          const active = to === '/' ? loc.pathname === '/' : loc.pathname.startsWith(to)
          return (
            <NavLink key={to} to={to} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '6px 14px', borderRadius: 6, fontSize: 13, fontWeight: 600,
              textDecoration: 'none', transition: 'all 0.15s',
              color: active ? '#fff' : '#94a3b8',
              background: active ? 'rgba(255,255,255,0.1)' : 'transparent',
              borderBottom: active ? '2px solid #818cf8' : '2px solid transparent',
            }}>
              <Icon size={14} />
              {label}
            </NavLink>
          )
        })}

        {/* Right side — coming soon */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6, opacity: 0.5 }}>
          <Shuffle size={13} color="#94a3b8" />
          <span style={{ fontSize: 12, color: '#94a3b8' }}>Trade Advisor</span>
          <span style={{ fontSize: 10, color: '#64748b', background: 'rgba(255,255,255,0.08)', padding: '2px 7px', borderRadius: 99, letterSpacing: '0.05em' }}>SOON</span>
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <main style={{ maxWidth: 1280, margin: '0 auto', padding: '70px 20px 48px' }}>
        <Routes>
          <Route path="/"             element={<Live />} />
          <Route path="/upcoming"     element={<Upcoming />} />
          <Route path="/injuries"     element={<Injuries />} />
          <Route path="/optimizer"    element={<LineupOptimizer />} />
          <Route path="/game/:gameId" element={<GameDetail />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
