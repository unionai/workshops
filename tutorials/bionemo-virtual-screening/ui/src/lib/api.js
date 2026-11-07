// Thin API client. Everything that talks to the backend goes through here so the
// base-URL logic lives in exactly one place.

// Empty by default: the API serves this bundle from the same origin, so every call is
// a plain relative path and inherits the page's session. VITE_API_BASE remains an
// escape hatch for pointing a local `npm run dev` at a remote API.
const BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '')

async function get(path) {
  const resp = await fetch(`${BASE}${path}`)
  if (!resp.ok) {
    throw new Error(`GET ${path} failed: ${resp.status} ${await resp.text()}`)
  }
  return resp.json()
}

async function post(path, body) {
  const resp = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) {
    let detail = await resp.text()
    try {
      detail = JSON.parse(detail).detail || detail
    } catch {
      // Non-JSON error body; use it as-is.
    }
    throw new Error(detail)
  }
  return resp.json()
}

export const api = {
  catalogue: () => get('/api/catalogue'),
  structure: (key) => get(`/api/structure/${key}`),
  launchScreen: (payload) => post('/api/screen', payload),
  runStatus: (name) => get(`/api/run/${name}`),
  runPose: (name, index) => get(`/api/run/${name}/pose/${index}`),
  leaderboard: (targetKey, limit = 20) =>
    get(`/api/leaderboard?limit=${limit}${targetKey ? `&target_key=${targetKey}` : ''}`),
  nims: () => get('/api/nims'),
}

// Poll a run until it finishes. `onPhase` fires on every phase change so the UI can
// narrate progress instead of showing a dead spinner.
export async function pollRun(runName, { onPhase, intervalMs = 1500, timeoutMs = 1800000 } = {}) {
  const deadline = Date.now() + timeoutMs
  let lastPhase = null

  while (Date.now() < deadline) {
    const status = await api.runStatus(runName)
    if (status.phase !== lastPhase) {
      lastPhase = status.phase
      onPhase?.(status)
    }
    if (status.done) return status
    await new Promise((r) => setTimeout(r, intervalMs))
  }

  throw new Error(`Run ${runName} did not finish within ${Math.round(timeoutMs / 1000)}s`)
}
