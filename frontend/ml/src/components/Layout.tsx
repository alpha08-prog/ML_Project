import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { Brain, BarChart3, LineChart, Zap, Info } from "lucide-react";
import ThemeToggle from "./ThemeToggle";

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();

  const NAV_ITEMS = [
    { path: "/", label: "Dashboard", icon: Brain },
    { path: "/performance", label: "Performance", icon: BarChart3 },
    { path: "/visualization", label: "Visualization", icon: LineChart },
    { path: "/predictions", label: "Predictions", icon: Zap },
    { path: "/about", label: "About", icon: Info },
  ];

  return (
    <div className="layout">
      <header className="header">
        <div className="header-content">
          <Link to="/" className="logo pulse hover-glow">
            <Brain className="logo-icon" />
            <span className="logo-text">EEG ML Project</span>
          </Link>
          <div className="flex items-center gap-sm">
            <nav className="nav">
              {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
                <Link
                  key={path}
                  to={path}
                  className={`nav-link ${location.pathname === path ? "active" : ""}`}
                >
                  <Icon size={18} />
                  <span>{label}</span>
                </Link>
              ))}
            </nav>
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main className="main-content">{children}</main>
    </div>
  );
}
