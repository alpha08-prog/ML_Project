import { cn } from '../../utils/cn'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  leftIcon?: ReactNode
  rightIcon?: ReactNode
}

export default function Button({
  className,
  variant = 'primary',
  size = 'md',
  leftIcon,
  rightIcon,
  children,
  ...props
}: ButtonProps) {
  const base = 'inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-all duration-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2'
  const variants = {
    primary:
      'bg-primary-600 text-white hover:bg-primary-700 shadow-sm hover:shadow-md focus-visible:outline-primary-600',
    secondary:
      'bg-white/70 dark:bg-slate-900/60 border border-slate-200/70 dark:border-slate-700/60 text-slate-700 dark:text-slate-200 hover:shadow-md focus-visible:outline-slate-400',
    ghost:
      'bg-transparent text-slate-700 dark:text-slate-200 hover:bg-slate-100/60 dark:hover:bg-slate-800/60 focus-visible:outline-slate-400',
  }
  const sizes = {
    sm: 'text-sm px-3 py-1.5',
    md: 'text-sm px-4 py-2',
    lg: 'text-base px-5 py-2.5',
  }

  return (
    <button className={cn(base, variants[variant], sizes[size], className)} {...props}>
      {leftIcon}
      {children}
      {rightIcon}
    </button>
  )
}
