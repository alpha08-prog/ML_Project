import { Moon, Sun } from 'lucide-react'
import { useTheme } from '../theme/ThemeProvider'

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  const isDark = theme === 'dark'

  return (
    <button
      onClick={toggleTheme}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="inline-flex items-center justify-center rounded-xl border border-slate-200/60 dark:border-slate-700/60 bg-white/70 dark:bg-slate-900/60 backdrop-blur px-3 py-2 text-slate-700 dark:text-slate-200 shadow-sm hover:shadow-md transition-all duration-200 hover:-translate-y-0.5"
    >
      {isDark ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  )
}
