import { type ReactNode } from 'react'
import { cn } from '../../utils/cn'

interface CardProps {
  className?: string
  children: ReactNode
}

export function Card({ className, children }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-2xl bg-white/80 dark:bg-slate-900/60 backdrop-blur p-6 shadow-soft hover:shadow-glass transition-all',
        className,
      )}
    >
      {children}
    </div>
  )
}

interface CardHeaderProps { className?: string; children: ReactNode }
export function CardHeader({ className, children }: CardHeaderProps) {
  return <div className={cn('mb-3', className)}>{children}</div>
}

interface CardTitleProps { className?: string; children: ReactNode }
export function CardTitle({ className, children }: CardTitleProps) {
  return <h3 className={cn('text-lg font-semibold text-slate-900 dark:text-slate-100', className)}>{children}</h3>
}

interface CardContentProps { className?: string; children: ReactNode }
export function CardContent({ className, children }: CardContentProps) {
  return <div className={cn('text-slate-600 dark:text-slate-300', className)}>{children}</div>
}
