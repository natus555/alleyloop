import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { Activity, Calendar, AlertTriangle, Users, TrendingUp, Shuffle, ListOrdered } from 'lucide-react'
import Live from './pages/Live'
import Upcoming from './pages/Upcoming'
import Injuries from './pages/Injuries'
import LineupOptimizer from './pages/LineupOptimizer'
import GameDetail from './pages/GameDetail'
import Rosters from './pages/Rosters'

const NAV = [
  { to: '/',          icon: Activity,       label: 'Live'      },
  { to: '/upcoming',  icon: Calendar,       label: 'Schedule'  },
  { to: '/injuries',  icon: AlertTriangle,  label: 'Injuries'  },
  { to: '/rosters',   icon: ListOrdered,    label: 'Rosters'   },
  { to: '/optimizer', icon: Users,          label: 'Optimizer' },
]

function NavBar() {
  const loc = useLocation()
  return (
    <nav className="fixed inset-x-0 top-0 z-50 border-b-2 border-indigo-500/70 bg-slate-950/90 shadow-lg shadow-slate-900/30 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-1 px-5">
        {/* Logo */}
        <div className="mr-6 flex shrink-0 items-center gap-2">
          <div className="flex size-7 items-center justify-center rounded-md bg-linear-to-br from-indigo-500 to-violet-600">
            <TrendingUp size={15} color="#fff" />
          </div>
          <span className="text-[17px] font-extrabold tracking-tight text-white">AlleyLoop</span>
        </div>

        {/* Nav links */}
        {NAV.map(({ to, icon: Icon, label }) => {
          const active = to === '/' ? loc.pathname === '/' : loc.pathname.startsWith(to)
          return (
            <NavLink
              key={to}
              to={to}
              className={`flex items-center gap-1.5 rounded-md border-b-2 px-3.5 py-1.5 text-[13px] font-semibold transition ${
                active
                  ? 'border-indigo-300 bg-white/10 text-white'
                  : 'border-transparent text-slate-400 hover:bg-white/5 hover:text-slate-200'
              }`}
            >
              <Icon size={14} />
              {label}
            </NavLink>
          )
        })}

        {/* Right side — coming soon */}
        <div className="ml-auto flex items-center gap-1.5 opacity-60">
          <Shuffle size={13} color="#94a3b8" />
          <span className="text-xs text-slate-400">Trade Advisor</span>
          <span className="rounded-full bg-white/10 px-1.5 py-0.5 text-[10px] tracking-wide text-slate-500">SOON</span>
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <main className="mx-auto max-w-7xl px-5 pb-12 pt-[72px]">
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
