import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ThemeContext, type ThemeContextValue } from './ThemeContext'

function getInitialTheme(): 'light' | 'dark' {
  return 'light'
}

function hasStoredTheme(): boolean {
  if (typeof window === 'undefined') return false

  try {
    const stored = localStorage.getItem('theme')
    return stored === 'light' || stored === 'dark'
  } catch {
    return false
  }
}

function persistTheme(theme: 'light' | 'dark') {
  if (typeof window === 'undefined') return

  try {
    localStorage.setItem('theme', theme)
  } catch {
    // Ignore storage failures so theme toggling still works.
  }
}

function applyTheme(theme: 'light' | 'dark') {
  const root = document.documentElement
  root.classList.toggle('dark', theme === 'dark')
  root.style.colorScheme = theme
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<'light' | 'dark'>(getInitialTheme)
  const hasStoredPreference = useRef(hasStoredTheme())

  useLayoutEffect(() => {
    applyTheme(theme)
  }, [theme])

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const handler = () => {
      if (!hasStoredPreference.current) {
        setThemeState(media.matches ? 'dark' : 'light')
      }
    }
    media.addEventListener('change', handler)
    return () => media.removeEventListener('change', handler)
  }, [])

  const value = useMemo<ThemeContextValue>(() => ({
    theme,
    toggleTheme: () => {
      hasStoredPreference.current = true
      const nextTheme = theme === 'light' ? 'dark' : 'light'
      setThemeState(nextTheme)
      persistTheme(nextTheme)
    },
    setTheme: (nextTheme) => {
      hasStoredPreference.current = true
      setThemeState(nextTheme)
      persistTheme(nextTheme)
    },
  }), [theme])

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}
