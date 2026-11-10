<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({
  catalogue: { type: Object, required: true },
  busy: { type: Boolean, default: false },
})

const emit = defineEmits(['run', 'reset'])

const targetKey = ref(null)
const seedKey = ref(null)
const seedSmiles = ref('')
const nickname = ref('')

// GenMol
const numMolecules = ref(10)
const temperature = ref(1.0)
const noise = ref(0.2)
const stepSize = ref(4)
const scoring = ref('QED')

// DiffDock
const numPoses = ref(10)
const timeDivisions = ref(20)
const numSteps = ref(18)

const showExamples = ref(false)

watch(
  () => props.catalogue,
  (cat) => {
    if (!cat) return
    const withStructure = cat.targets.find((t) => t.has_structure)
    targetKey.value = (withStructure || cat.targets[0])?.key ?? null
    const seed = cat.seeds[0]
    seedKey.value = seed?.key ?? null
    seedSmiles.value = seed?.smiles ?? ''
    numMolecules.value = cat.defaults.num_molecules
    temperature.value = cat.defaults.temperature
    noise.value = cat.defaults.noise
    stepSize.value = cat.defaults.step_size
    numPoses.value = cat.defaults.num_poses
  },
  { immediate: true },
)

// Picking a named seed overwrites the textarea; editing the textarea afterwards is
// allowed and simply detaches it from the preset.
watch(seedKey, (key) => {
  const seed = props.catalogue?.seeds.find((s) => s.key === key)
  if (seed) seedSmiles.value = seed.smiles
})

const target = computed(() => props.catalogue?.targets.find((t) => t.key === targetKey.value) ?? null)
const seed = computed(() => props.catalogue?.seeds.find((s) => s.key === seedKey.value) ?? null)

const mustFoldLive = computed(() => target.value != null && !target.value.has_structure)
const estimate = computed(() => (mustFoldLive.value ? 'several minutes' : '30–45 seconds'))

function run() {
  if (!target.value || !seedSmiles.value.trim()) return
  emit('run', {
    target_key: target.value.key,
    seed_smiles: seedSmiles.value.trim(),
    nickname: nickname.value.trim() || 'anonymous',
    num_molecules: numMolecules.value,
    temperature: temperature.value,
    noise: noise.value,
    step_size: stepSize.value,
    scoring: scoring.value,
    num_poses: numPoses.value,
    fold_live: mustFoldLive.value,
  })
}

function reset() {
  const cat = props.catalogue
  const s = cat.seeds[0]
  seedKey.value = s?.key ?? null
  seedSmiles.value = s?.smiles ?? ''
  numMolecules.value = cat.defaults.num_molecules
  temperature.value = cat.defaults.temperature
  noise.value = cat.defaults.noise
  stepSize.value = cat.defaults.step_size
  numPoses.value = cat.defaults.num_poses
  timeDivisions.value = 20
  numSteps.value = 18
  emit('reset')
}
</script>

<template>
  <div class="input-col">
    <div class="col-head">
      <h2>Input</h2>
      <button class="link-btn" @click="showExamples = !showExamples">
        {{ showExamples ? 'Hide' : 'View' }} Examples
      </button>
    </div>

    <div v-if="showExamples" class="examples panel-card">
      <p>Pick a target and a seed, then run. Three starting points worth trying:</p>
      <ul>
        <li><strong>Blueprint scaffold</strong> on SARS-CoV-2 Mpro — matches NVIDIA's reference run.</li>
        <li><strong>De novo (small)</strong> on CDK2 — unconstrained generation against a kinase.</li>
        <li>Raise <strong>temperature</strong> to 2.0 for more adventurous, less drug-like molecules.</li>
      </ul>
    </div>

    <!-- Receptor -->
    <section class="panel-card">
      <header class="card-head">
        <span class="nim-tag nim-tag-static">RECEPTOR</span>
        <h3>Target protein</h3>
      </header>
      <p class="card-note">
        Experimental crystal structures from the RCSB PDB, solved to sub-&aring;ngstr&ouml;m
        resolution.
      </p>

      <label for="target">Protein Examples</label>
      <select id="target" v-model="targetKey" :disabled="busy">
        <option v-for="t in catalogue.targets" :key="t.key" :value="t.key">
          {{ t.label }}{{ t.has_structure ? '' : '  (no cached structure)' }}
        </option>
      </select>

      <div v-if="target" class="meta-grid">
        <div><span>Length</span><strong>{{ target.length }} aa</strong></div>
        <div><span>Source</span><strong class="mono">{{ target.source }}</strong></div>
      </div>
      <p v-if="target" class="hint">{{ target.blurb }}</p>
    </section>

    <!-- GenMol -->
    <section class="panel-card">
      <header class="card-head">
        <span class="nim-tag">GENMOL</span>
        <h3>Molecule generation</h3>
      </header>
      <p class="card-note">
        A masked diffusion model over SAFE fragment strings. The <code>[*{'{'}m-n{'}'}]</code> mask says how
        many atoms it may add.
      </p>

      <label for="seed">Molecule Examples</label>
      <select id="seed" v-model="seedKey" :disabled="busy">
        <option v-for="s in catalogue.seeds" :key="s.key" :value="s.key">{{ s.label }}</option>
      </select>
      <p v-if="seed" class="hint">{{ seed.blurb }}</p>

      <label for="seed-smiles">Molecule Sequence <span class="req">*</span></label>
      <textarea id="seed-smiles" v-model="seedSmiles" rows="2" class="mono" :disabled="busy" />

      <div class="slider-row">
        <label>Molecules to generate</label>
        <input v-model.number="numMolecules" type="range" min="1" max="30" step="1" :disabled="busy" />
        <output>{{ numMolecules }}</output>
      </div>
      <div class="slider-row">
        <label>Temperature</label>
        <input v-model.number="temperature" type="range" min="0.1" max="3" step="0.1" :disabled="busy" />
        <output>{{ temperature.toFixed(1) }}</output>
      </div>
      <div class="slider-row">
        <label>Noise</label>
        <input v-model.number="noise" type="range" min="0" max="1" step="0.05" :disabled="busy" />
        <output>{{ noise.toFixed(2) }}</output>
      </div>
      <div class="slider-row">
        <label>Step size</label>
        <input v-model.number="stepSize" type="range" min="1" max="16" step="1" :disabled="busy" />
        <output>{{ stepSize }}</output>
      </div>

      <label for="scoring">Property to Compute</label>
      <select id="scoring" v-model="scoring" :disabled="busy">
        <option value="QED">QED — drug-likeness</option>
      </select>
    </section>

    <!-- DiffDock -->
    <section class="panel-card">
      <header class="card-head">
        <span class="nim-tag">DIFFDOCK</span>
        <h3>Protein-ligand docking</h3>
      </header>
      <p class="card-note">
        Blind docking: no binding pocket is supplied. Every generated molecule is docked against the
        receptor in one batched request.
      </p>

      <div class="slider-row">
        <label>Poses per molecule</label>
        <input v-model.number="numPoses" type="range" min="1" max="30" step="1" :disabled="busy" />
        <output>{{ numPoses }}</output>
      </div>
      <div class="slider-row">
        <label>Time divisions</label>
        <input v-model.number="timeDivisions" type="range" min="5" max="40" step="1" :disabled="busy" />
        <output>{{ timeDivisions }}</output>
      </div>
      <div class="slider-row">
        <label>Diffusion steps</label>
        <input v-model.number="numSteps" type="range" min="5" max="40" step="1" :disabled="busy" />
        <output>{{ numSteps }}</output>
      </div>
    </section>

    <!-- Submit -->
    <section class="panel-card submit-card">
      <label for="nickname">Your name for the leaderboard</label>
      <input id="nickname" v-model="nickname" type="text" maxlength="32" placeholder="anonymous" :disabled="busy" />

      <p class="terms">
        Runs execute on Union as Flyte&nbsp;2 workflows against self-hosted NVIDIA NIM
        microservices. Each run returns a link to its execution graph.
      </p>

      <div class="actions">
        <button class="btn-ghost" :disabled="busy" @click="reset">Reset</button>
        <button class="btn-primary" :disabled="busy || !target || !seedSmiles.trim()" @click="run">
          {{ busy ? 'Generating…' : 'Generate Molecules' }}
        </button>
      </div>
      <p class="hint estimate">Expect {{ estimate }}.</p>
    </section>
  </div>
</template>

<style scoped>
.input-col {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}

.col-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  padding-bottom: 6px;
}

.col-head h2 {
  font-size: 18px;
}

.link-btn {
  background: none;
  color: var(--accent);
  font-size: 13px;
  font-weight: 600;
  padding: 0;
}

.link-btn:hover {
  text-decoration: underline;
}

.panel-card {
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
}

.card-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.card-head h3 {
  font-size: 15px;
}

.nim-tag {
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--accent);
  border: 1px solid var(--accent-dim);
  border-radius: 4px;
  padding: 2px 6px;
}

.nim-tag-static {
  color: var(--text-dim);
  border-color: var(--border-bright);
}

.card-note {
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.5;
  margin: 0 0 14px;
}

.examples {
  font-size: 13px;
  color: var(--text-dim);
}

.examples ul {
  margin: 8px 0 0;
  padding-left: 18px;
  line-height: 1.7;
}

.req {
  color: var(--error);
}

textarea {
  width: 100%;
  background: var(--bg-input);
  border: 1px solid var(--border-bright);
  border-radius: 8px;
  color: var(--text);
  padding: 9px 11px;
  font-size: 12px;
  resize: vertical;
  margin-bottom: 14px;
}

textarea:focus {
  outline: 2px solid var(--accent-dim);
}

.meta-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 14px;
  margin: 12px 0 6px;
  font-size: 12px;
}

.meta-grid div {
  display: flex;
  justify-content: space-between;
  gap: 8px;
}

.meta-grid span {
  color: var(--text-dim);
}

.meta-grid strong {
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.slider-row {
  display: grid;
  grid-template-columns: 1fr 130px 44px;
  align-items: center;
  gap: 10px;
  margin-bottom: 9px;
}

.slider-row label {
  margin: 0;
  font-size: 12px;
  font-weight: 500;
  color: var(--text);
}

.slider-row output {
  font-family: var(--mono);
  font-size: 12px;
  text-align: right;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}

/* Deliberately not sticky: sticky positioning lifted this card out of document order
   and made it overlap the DiffDock card above it. */
.submit-card {
  border-color: var(--border-bright);
}

.terms {
  font-size: 11px;
  color: var(--text-dim);
  line-height: 1.5;
  margin: 14px 0;
}

.actions {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 10px;
}

.estimate {
  text-align: center;
  margin-top: 8px;
}
</style>
