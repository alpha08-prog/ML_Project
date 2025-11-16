import { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Brain, BarChart3, LineChart, Zap, Info } from 'lucide-react'
import './Layout.css'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Brain },
    { path: '/performance', label: 'Performance', icon: BarChart3 },
    { path: '/visualization', label: 'Visualization', icon: LineChart },
    { path: '/predictions', label: 'Predictions', icon: Zap },
    { path: '/about', label: 'About', icon: Info },
  ]

  return (
    <div className="layout">
      <header className="header">
        <div className="header-content">
          <Link to="/" className="logo">
            <Brain className="logo-icon" />
            <span className="logo-text">EEG ML Project</span>
          </Link>
          <nav className="nav">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`nav-link ${isActive ? 'active' : ''}`}
                >
                  <Icon size={18} />
                  <span>{item.label}</span>
                </Link>
              )
            })}
          </nav>
        </div>
      </header>
      <main className="main-content">
        {children}
      </main>
    </div>
  )
}

