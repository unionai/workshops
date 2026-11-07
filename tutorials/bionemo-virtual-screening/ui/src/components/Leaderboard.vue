<script setup>
import MoleculeImage from './MoleculeImage.vue'
import { confidenceTier, formatConfidence } from '../lib/confidence'

const props = defineProps({
  entries: { type: Array, required: true },
  stats: { type: Object, default: () => ({}) },
  targets: { type: Array, default: () => [] },
  filterKey: { type: String, default: null },
  // The run currently shown in the output panel, and which molecule within it.
  highlightRun: { type: String, default: null },
  selectedSmiles: { type: String, default: null },
})

const emit = defineEmits(['filter', 'open'])

function fmt(value, digits = 3) {
  return value == null ? '—' : Number(value).toFixed(digits)
}
</script>

<template>
  <div class="panel">
    <div class="lb-head">
      <div class="panel-title" style="margin: 0">Leaderboard</div>
      <select class="lb-filter" :value="filterKey ?? ''" @change="emit('filter', $event.target.value || null)">
        <option value="">All targets</option>
        <option v-for="t in targets" :key="t.key" :value="t.key">{{ t.label }}</option>
      </select>
    </div>

    <div class="stats">
      <div class="stat">
        <strong>{{ stats.runs ?? 0 }}</strong>
        <span>runs</span>
      </div>
      <div class="stat">
        <strong>{{ stats.molecules ?? 0 }}</strong>
        <span>molecules</span>
      </div>
      <div class="stat">
        <strong>{{ stats.players ?? 0 }}</strong>
        <span>players</span>
      </div>
      <div class="stat">
        <strong>{{ fmt(stats.best_pose_confidence) }}</strong>
        <span>best conf.</span>
      </div>
    </div>

    <div v-if="!entries.length" class="empty">
      Nothing on the board yet. Run the first screen.
    </div>

    <ol v-else class="rows">
      <li
        v-for="e in entries"
        :key="e.run_name + e.smiles"
        class="row"
        :class="{
          showing: e.run_name === highlightRun && e.smiles === selectedSmiles,
          podium: e.rank <= 3,
        }"
        :title="`Show this run's molecules and poses`"
        @click="emit('open', e)"
      >
        <div class="rank">{{ e.rank }}</div>
        <MoleculeImage :smiles="e.smiles" :width="86" :height="64" />
        <div class="row-body">
          <div class="row-top">
            <span class="who">{{ e.nickname }}</span>
            <span class="conf" :class="`tier-${confidenceTier(e.pose_conf)}`">
              {{ formatConfidence(e.pose_conf) }}
            </span>
          </div>
          <div class="row-bottom">
            <span class="target">{{ e.target_label }}</span>
            <span class="sep">·</span>
            <span>QED {{ fmt(e.qed, 2) }}</span>
            <!-- stop propagation so the external link doesn't also re-open the run
                 in the output panel behind the new tab -->
            <a :href="e.run_url" target="_blank" rel="noopener" class="run" @click.stop>
              union&nbsp;&rarr;
            </a>
          </div>
        </div>
      </li>
    </ol>
  </div>
</template>

<style scoped>
.lb-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
}

.lb-filter {
  width: auto;
  max-width: 180px;
  font-size: 13px;
  padding: 6px 10px;
}

.stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin-bottom: 18px;
}

.stat {
  background: var(--bg-input);
  border-radius: 8px;
  padding: 10px 6px;
  text-align: center;
}

.stat strong {
  display: block;
  font-size: 19px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}

.stat span {
  font-size: 10px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.empty {
  color: var(--text-dim);
  font-size: 14px;
  text-align: center;
  padding: 26px 0;
}

.rows {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 7px;
  max-height: 560px;
  overflow-y: auto;
}

.row {
  display: flex;
  align-items: center;
  gap: 11px;
  background: var(--bg-input);
  border: 1px solid transparent;
  border-radius: 9px;
  padding: 9px 11px;
  cursor: pointer;
  transition: border-color 120ms ease, background 120ms ease;
}

.row:hover {
  border-color: var(--accent);
  background: #1f252c;
}

.row.podium .rank {
  color: var(--accent);
}

/* Exactly one highlighted row: the molecule currently on screen.
   An earlier version also tinted every row from the same run, which meant clicking
   one row lit up two — indistinguishable from a bug. One selection, one highlight,
   the same way the molecule chips and pose ranks behave. */
.row.showing {
  border-color: var(--accent);
  background: rgba(118, 185, 0, 0.14);
}

.row.showing .rank {
  color: var(--accent);
}

.rank {
  font-size: 15px;
  font-weight: 800;
  color: var(--text-dim);
  width: 22px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  flex-shrink: 0;
}

.row-body {
  min-width: 0;
  flex: 1;
}

.row-top {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}

.who {
  font-weight: 650;
  font-size: 14px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conf {
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  flex-shrink: 0;
}

.tier-high {
  color: var(--accent);
}

.tier-moderate {
  color: var(--warn);
}

.tier-low {
  color: var(--error);
}

.tier-none {
  color: var(--text-dim);
}

.row-bottom {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 2px;
}

.target {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.sep {
  opacity: 0.5;
}

.run {
  margin-left: auto;
  text-decoration: none;
  flex-shrink: 0;
  font-weight: 600;
}

.run:hover {
  text-decoration: underline;
}
</style>
