interface Props { tri: string; color?: string; size?: number; logo?: string }

export default function TeamBadge({ tri, color = '#4f46e5', size = 36, logo }: Props) {
  if (logo) {
    return (
      <div style={{
        width: size, height: size, borderRadius: size * 0.22,
        background: `${color}10`,
        border: `2px solid ${color}30`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0, overflow: 'hidden', padding: 3,
      }}>
        <img
          src={logo}
          alt={tri}
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          onError={e => {
            // Fallback to text if logo fails to load
            const parent = (e.target as HTMLImageElement).parentElement
            if (parent) {
              parent.innerHTML = `<span style="font-weight:900;font-size:${size * 0.3}px;color:${color};letter-spacing:-0.5px">${tri}</span>`
            }
          }}
        />
      </div>
    )
  }

  return (
    <div style={{
      width: size, height: size, borderRadius: size * 0.22,
      background: `${color}15`,
      border: `2px solid ${color}40`,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontWeight: 900, fontSize: size * 0.3, color,
      flexShrink: 0, letterSpacing: '-0.5px',
    }}>
      {tri}
    </div>
  )
}
