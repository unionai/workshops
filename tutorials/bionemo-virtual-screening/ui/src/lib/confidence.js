// Interpreting DiffDock's confidence score.
//
// The confidence model outputs an unbounded logit, in practice roughly -7 to +3.
// The convention from the DiffDock paper and NVIDIA's docs is:
//
//   c >  0     high confidence in the pose
//  -1.5 < c< 0 moderate
//   c < -1.5   low
//
// This matters for the booth: rendering -0.2 in the same green as 0.95 implies a
// good result where there isn't one, and this audience reads docking output for a
// living. Colour by tier instead.

export function confidenceTier(value) {
  if (value == null) return 'none'
  if (value > 0) return 'high'
  if (value > -1.5) return 'moderate'
  return 'low'
}

export const TIER_LABEL = {
  high: 'high confidence',
  moderate: 'moderate',
  low: 'low confidence',
  none: 'not docked',
}

export function formatConfidence(value, digits = 3) {
  return value == null ? '—' : Number(value).toFixed(digits)
}
