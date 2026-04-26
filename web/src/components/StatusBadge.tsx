interface Props { status: string; small?: boolean }

function getColors(s: string): { fg: string; bg: string; border: string } {
  const l = s.toLowerCase()
  if (l.includes('out') || l.includes('suspension') || l.includes('injured reserve'))
    return { fg: '#dc2626', bg: '#fee2e2', border: '#fca5a5' }
  if (l.includes('doubtful'))
    return { fg: '#c2410c', bg: '#ffedd5', border: '#fdba74' }
  if (l.includes('questionable') || l.includes('game time') || l.includes('gtd'))
    return { fg: '#d97706', bg: '#fef3c7', border: '#fde68a' }
  if (l.includes('day-to-day') || l.includes('day to day') || l.includes('probable'))
    return { fg: '#1d4ed8', bg: '#dbeafe', border: '#93c5fd' }
  return { fg: '#15803d', bg: '#dcfce7', border: '#86efac' }
}

export default function StatusBadge({ status, small }: Props) {
  const { fg, bg, border } = getColors(status)
  return (
    <span style={{
      display: 'inline-block',
      padding: small ? '1px 6px' : '3px 9px',
      borderRadius: 99,
      fontSize: small ? 10 : 11,
      fontWeight: 700,
      color: fg,
      background: bg,
      border: `1px solid ${border}`,
      whiteSpace: 'nowrap',
    }}>
      {status}
    </span>
  )
}
