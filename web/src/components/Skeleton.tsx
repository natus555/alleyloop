interface Props { width?: string | number; height?: number; className?: string }

export default function Skeleton({ width = '100%', height = 16 }: Props) {
  return (
    <div className="shimmer" style={{ width, height, borderRadius: 6 }} />
  )
}
