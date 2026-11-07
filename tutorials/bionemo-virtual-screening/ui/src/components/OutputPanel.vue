<script setup>
import { computed, ref, watch } from 'vue'
import MoleculeImage from './MoleculeImage.vue'
import PoseViewer from './PoseViewer.vue'
import { TIER_LABEL, confidenceTier, formatConfidence } from '../lib/confidence'

const props = defineProps({
  candidates: { type: Array, default: () => [] },
  selectedIndex: { type: Number, default: null },
  poseData: { type: Object, default: null },
  proteinPdb: { type: String, default: null },
  poseLoading: { type: Boolean, default: false },
  run: { type: Object, default: null },
  busy: { type: Boolean, default: false },
})

const emit = defineEmits(['select', 'select-pose'])

const tab = ref('preview')
const poseIndex = ref(0)

// Reset to the best pose whenever a different molecule is selected, otherwise the
// viewer silently keeps showing rank 4 of the previous molecule.
watch(() => props.selectedIndex, () => { poseIndex.value = 0 })

const selected = computed(() =>
  props.selectedIndex != null ? props.candidates[props.selectedIndex] ?? null : null,
)

const poses = computed(() => props.poseData?.poses ?? [])
const confidences = computed(() => props.poseData?.confidences ?? [])

const activePose = computed(() => poses.value[poseIndex.value] ?? props.poseData?.pose ?? null)
const activeConfidence = computed(() =>
  confidences.value[poseIndex.value] ?? props.poseData?.pose_confidence ?? null,
)

// The JSON tab deliberately shows exactly what the pipeline returned, minus the pose
// blobs. People at a booth ask "can I get this out", and showing the payload answers it.
const jsonView = computed(() =>
  JSON.stringify(
    {
      run: props.run?.run_name,
      url: props.run?.url,
      target: props.run?.target_label,
      timings_seconds: props.run?.timings,
      candidates: props.candidates,
    },
    null,
    2,
  ),
)

function download() {
  const blob = new Blob([jsonView.value], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `${props.run?.run_name || 'screening'}.json`
  a.click()
  URL.revokeObjectURL(a.href)
}

const hasResults = computed(() => props.candidates.length > 0)
</script>

<template>
  <div class="output-col">
    <div class="col-head">
      <h2>Output</h2>
      <div class="head-right">
        <div class="tabs">
          <button :class="{ on: tab === 'preview' }" @click="tab = 'preview'">Preview</button>
          <button :class="{ on: tab === 'json' }" @click="tab = 'json'">JSON</button>
        </div>
        <button class="link-btn" :disabled="!hasResults" @click="download">Download</button>
      </div>
    </div>

    <!-- Empty / running -->
    <div v-if="!hasResults" class="panel-card empty-state">
      <template v-if="busy">
        <div class="spinner-lg" />
        <p>Running on Union…</p>
        <a v-if="run?.url" :href="run.url" target="_blank" rel="noopener">Watch the execution graph&nbsp;&rarr;</a>
      </template>
      <template v-else>
        <p>Set up a screen and hit <strong>Generate Molecules</strong>.</p>
        <p class="sub">Generated molecules, their docked poses and a ranked leaderboard appear here.</p>
      </template>
    </div>

    <template v-else-if="tab === 'preview'">
      <!-- Molecule strip -->
      <div class="panel-card">
        <div class="strip-head">
          <span>{{ candidates.length }} molecules generated</span>
          <span class="dim">ranked by pose confidence</span>
        </div>
        <div class="strip">
          <button
            v-for="(c, i) in candidates"
            :key="c.smiles + i"
            class="chip"
            :class="{ on: selectedIndex === i, dud: !c.pose_confidence }"
            @click="emit('select', i)"
          >
            <MoleculeImage :smiles="c.smiles" :width="128" :height="92" />
            <span class="chip-score" :class="`tier-${confidenceTier(c.pose_confidence)}`">
              {{ formatConfidence(c.pose_confidence, 2) }}
            </span>
          </button>
        </div>
      </div>

      <!-- Viewer + pose ranking -->
      <div class="viewer-row">
        <PoseViewer
          class="viewer"
          :protein-pdb="proteinPdb"
          :pose="activePose"
          :smiles="selected?.smiles"
          :confidence="activeConfidence"
          :loading="poseLoading"
        />

        <div class="panel-card ranks">
          <div class="panel-title" style="margin-bottom: 10px">Poses</div>
          <p v-if="!poses.length" class="dim small">Select a molecule to see its poses.</p>
          <label v-for="(p, i) in poses" :key="i" class="rank-row" :class="{ on: poseIndex === i }">
            <input type="radio" :value="i" :checked="poseIndex === i" @change="poseIndex = i" />
            <span class="rank-label">Rank {{ i + 1 }}</span>
            <span class="rank-score" :class="`tier-${confidenceTier(confidences[i])}`">
              {{ formatConfidence(confidences[i]) }}
            </span>
          </label>

          <div v-if="selected" class="props">
            <div><span>QED</span><strong>{{ (selected.qed ?? 0).toFixed(2) }}</strong></div>
            <div><span>MW</span><strong>{{ (selected.molecular_weight ?? 0).toFixed(0) }}</strong></div>
            <div><span>logP</span><strong>{{ (selected.logp ?? 0).toFixed(1) }}</strong></div>
            <div><span>Rings</span><strong>{{ selected.ring_count ?? '—' }}</strong></div>
            <div><span>Rot. bonds</span><strong>{{ selected.num_rotatable_bonds ?? '—' }}</strong></div>
            <div><span>Heavy atoms</span><strong>{{ selected.num_heavy_atoms ?? '—' }}</strong></div>
          </div>
        </div>
      </div>

      <div class="caveat">
        <strong>Pose confidence</strong> is DiffDock's estimate that the predicted geometry is correct
        — currently <strong>{{ TIER_LABEL[confidenceTier(activeConfidence)] }}</strong> for this pose.
        It is not a binding affinity, and DiffDock does not predict one.
      </div>
    </template>

    <!-- JSON -->
    <div v-else class="panel-card json-card">
      <pre>{{ jsonView }}</pre>
    </div>
  </div>
</template>

<style scoped>
.output-col {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}

.col-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 6px;
}

.col-head h2 {
  font-size: 18px;
}

.head-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.tabs {
  display: flex;
  gap: 2px;
  background: var(--bg-input);
  border-radius: 8px;
  padding: 3px;
}

.tabs button {
  background: none;
  color: var(--text-dim);
  font-size: 13px;
  font-weight: 600;
  padding: 5px 14px;
  border-radius: 6px;
}

.tabs button.on {
  background: var(--bg-raised);
  color: var(--text);
}

.link-btn {
  background: none;
  color: var(--accent);
  font-size: 13px;
  font-weight: 600;
  padding: 0;
}

.link-btn:disabled {
  color: var(--text-dim);
}

.panel-card {
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  min-height: 320px;
  color: var(--text-dim);
  text-align: center;
}

.empty-state .sub {
  font-size: 13px;
  max-width: 420px;
}

.spinner-lg {
  width: 30px;
  height: 30px;
  border: 3px solid var(--accent);
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 800ms linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.strip-head {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 12px;
}

.dim {
  color: var(--text-dim);
  font-weight: 400;
}

.small {
  font-size: 12px;
}

.strip {
  display: flex;
  gap: 10px;
  overflow-x: auto;
  padding-bottom: 6px;
}

.chip {
  position: relative;
  flex-shrink: 0;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 9px;
  padding: 7px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 5px;
}

.chip.on {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent);
}

.chip.dud {
  opacity: 0.5;
}

.chip-score {
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.viewer-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 215px;
  gap: 14px;
  align-items: start;
}

.ranks {
  max-height: 520px;
  overflow-y: auto;
}

.rank-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 7px;
  border-radius: 6px;
  cursor: pointer;
  margin: 0;
  font-size: 12px;
}

.rank-row:hover {
  background: var(--bg-input);
}

.rank-row.on {
  background: var(--bg-input);
}

.rank-row input {
  accent-color: var(--accent);
  cursor: pointer;
}

.rank-label {
  flex: 1;
  color: var(--text);
  font-weight: 500;
}

.rank-score {
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}

.props {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
}

.props div {
  display: flex;
  justify-content: space-between;
}

.props span {
  color: var(--text-dim);
}

.props strong {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}

.caveat {
  background: rgba(240, 180, 0, 0.09);
  border-left: 3px solid var(--warn);
  padding: 10px 14px;
  border-radius: 0 8px 8px 0;
  font-size: 12.5px;
  color: #d8cfae;
  line-height: 1.55;
}

.json-card pre {
  margin: 0;
  font-family: var(--mono);
  font-size: 11.5px;
  line-height: 1.55;
  color: var(--text-dim);
  max-height: 620px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
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

@media (max-width: 1100px) {
  .viewer-row {
    grid-template-columns: 1fr;
  }
  .ranks {
    max-height: none;
  }
}
</style>
