<script setup>
import { computed } from 'vue'

const props = defineProps({
  // null when nothing has been launched yet.
  run: { type: Object, default: null },
  foldLive: { type: Boolean, default: false },
  error: { type: String, default: null },
})

// The stages a visitor should see. With a precomputed structure the fold stage is
// skipped entirely, so it isn't shown.
const stages = computed(() => {
  const base = [
    { key: 'generate', label: 'GenMol', detail: 'generating molecules' },
    { key: 'dock', label: 'DiffDock', detail: 'docking candidates' },
  ]
  return props.foldLive
    ? [{ key: 'fold', label: 'MSA + OpenFold2', detail: 'folding the target' }, ...base]
    : base
})

const timings = computed(() => props.run?.timings ?? {})
const done = computed(() => props.run?.done === true)
const phase = computed(() => props.run?.phase ?? 'QUEUED')

function stageState(key) {
  if (timings.value[key] != null) return 'done'
  if (done.value) return 'skipped'
  // Stages complete in order, so the first one without a timing is the active one.
  const idx = stages.value.findIndex((s) => timings.value[s.key] == null)
  return stages.value[idx]?.key === key ? 'active' : 'pending'
}

const phaseClass = computed(() => {
  if (props.error) return 'pill-error'
  if (phase.value === 'SUCCEEDED') return 'pill-ok'
  if (phase.value === 'FAILED' || phase.value === 'ABORTED') return 'pill-error'
  return 'pill-warn'
})

const totalTime = computed(() => {
  const vals = Object.values(timings.value)
  return vals.length ? vals.reduce((a, b) => a + b, 0).toFixed(1) : null
})
</script>

<template>
  <div class="panel">
    <div class="panel-title">Run</div>

    <div v-if="!run && !error" class="idle">
      Pick a target and hit <strong>Run screening</strong>. Every click is a real Flyte run on Union — you'll get
      a link to its graph.
    </div>

    <div v-if="error" class="error-box">{{ error }}</div>

    <template v-if="run">
      <div class="run-head">
        <span class="pill" :class="phaseClass">{{ phase }}</span>
        <a v-if="run.url" :href="run.url" target="_blank" rel="noopener" class="run-link">
          Open in Union&nbsp;&rarr;
        </a>
      </div>

      <div class="stages">
        <div v-for="s in stages" :key="s.key" class="stage" :class="stageState(s.key)">
          <div class="stage-dot">
            <span v-if="stageState(s.key) === 'done'">&check;</span>
            <span v-else-if="stageState(s.key) === 'active'" class="spinner" />
          </div>
          <div class="stage-body">
            <div class="stage-label">{{ s.label }}</div>
            <div class="stage-detail">
              <template v-if="timings[s.key] != null">{{ timings[s.key] }}s</template>
              <template v-else-if="stageState(s.key) === 'active'">{{ s.detail }}…</template>
              <template v-else>{{ s.detail }}</template>
            </div>
          </div>
        </div>
      </div>

      <div v-if="totalTime" class="total">
        Total compute <strong>{{ totalTime }}s</strong>
      </div>

      <div v-if="run.error" class="error-box">{{ run.error }}</div>
    </template>
  </div>
</template>

<style scoped>
.idle {
  color: var(--text-dim);
  font-size: 14px;
  line-height: 1.6;
}

.run-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
}

.run-link {
  font-size: 13px;
  font-weight: 600;
  text-decoration: none;
}

.run-link:hover {
  text-decoration: underline;
}

.stages {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stage {
  display: flex;
  gap: 13px;
  align-items: flex-start;
  padding: 9px 0;
}

.stage-dot {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: 2px solid var(--border-bright);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 12px;
  font-weight: 700;
  margin-top: 1px;
}

.stage.done .stage-dot {
  background: var(--accent);
  border-color: var(--accent);
  color: var(--accent-text);
}

.stage.active .stage-dot {
  border-color: var(--accent);
}

.stage-label {
  font-weight: 600;
  font-size: 14px;
}

.stage.pending .stage-label,
.stage.skipped .stage-label {
  color: var(--text-dim);
}

.stage-detail {
  font-size: 12px;
  color: var(--text-dim);
}

.spinner {
  width: 9px;
  height: 9px;
  border: 2px solid var(--accent);
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 700ms linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.total {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--border);
  font-size: 14px;
  color: var(--text-dim);
}

.total strong {
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}

.error-box {
  background: rgba(255, 92, 92, 0.1);
  border-left: 3px solid var(--error);
  padding: 11px 14px;
  border-radius: 0 8px 8px 0;
  font-size: 13px;
  color: #ffbdbd;
  margin-top: 14px;
}
</style>
