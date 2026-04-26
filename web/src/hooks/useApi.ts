import { useState, useEffect, useCallback } from 'react'

const BASE = '/api'

export function useApi<T>(
  path: string | null,
  deps: unknown[] = [],
  pollMs?: number,
) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetch_ = useCallback(async () => {
    if (!path) return
    setLoading(true)
    try {
      const r = await fetch(BASE + path)
      if (!r.ok) throw new Error(`${r.status}`)
      setData(await r.json())
      setError(null)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [path, ...deps])

  useEffect(() => { fetch_() }, [fetch_])

  useEffect(() => {
    if (!pollMs || pollMs <= 0) return
    const id = setInterval(fetch_, pollMs)
    return () => clearInterval(id)
  }, [fetch_, pollMs])

  return { data, loading, error, refetch: fetch_ }
}

export async function apiFetch<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path)
  if (!r.ok) throw new Error(`${r.status}`)
  return r.json()
}
