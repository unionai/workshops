<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { api, pollRun } from './lib/api'
import InputPanel from './components/InputPanel.vue'
import OutputPanel from './components/OutputPanel.vue'
import Leaderboard from './components/Leaderboard.vue'
import RunProgress from './components/RunProgress.vue'

const catalogue = ref(null)
const bootError = ref(null)

const run = ref(null)
const busy = ref(false)
const runError = ref(null)
const foldLive = ref(false)

const candidates = ref([])
const selectedIndex = ref(null)
const proteinPdb = ref(null)
const poseData = ref(null)
const poseLoading = ref(false)

const leaderboard = ref({ entries: [], stats: {} })
const leaderboardFilter = ref(null)
const nimStatus = ref([])
const showLeaderboard = ref(true)

let timer = null

// Only the two booth NIMs matter for readiness; MSA/OpenFold2 are intentionally not
// deployed, so counting them would permanently show the demo as degraded.
const BOOTH_NIMS = ['nim-genmol', 'nim-diffdock']
const boothNims = computed(() => nimStatus.value.filter((n) => BOOTH_NIMS.includes(n.name)))
const nimsReady = computed(() => boothNims.value.length > 0 && boothNims.value.every((n) => n.status === 'ready'))

async function loadCatalogue() {
  try {
    catalogue.value = await api.catalogue()
  } catch (err) {
    bootError.value = `Could not reach the API: ${err.message}`
  }
}

async function loadLeaderboard() {
  try {
    leaderboard.value = await api.leaderboard(leaderboardFilter.value, 20)
  } catch {
    /* a failed refresh is not worth interrupting the demo for */
  }
}

async function loadNims() {
  try {
    nimStatus.value = (await api.nims()).nims
  } catch {
    nimStatus.value = []
  }
}

async function onRun(request) {
  busy.value = true
  runError.value = null
  run.value = null
  candidates.value = []
  selectedIndex.value = null
  poseData.value = null
  foldLive.value = request.fold_live

  try {
    const launched = await api.launchScreen(request)
    run.value = { ...launched, phase: 'QUEUED', done: false, timings: {} }

    // Show the receptor immediately so something is on screen during the run.
    if (!request.fold_live) {
      api.structure(request.target_key).then((s) => { proteinPdb.value = s.pdb }).catch(() => {})
    }

    const final = await pollRun(launched.run_name, {
      onPhase: (status) => { run.value = { ...run.value, ...status } },
    })
    run.value = { ...run.value, ...final }

    if (final.error) {
      runError.value = final.error
    } else {
      candidates.value = final.candidates ?? []
      if (candidates.value.length) await selectCandidate(0)
      await loadLeaderboard()
    }
  } catch (err) {
    runError.value = err.message
  } finally {
    busy.value = false
  }
}

async function selectCandidate(index) {
  selectedIndex.value = index
  poseLoading.value = true
  try {
    const data = await api.runPose(run.value.run_name, index)
    poseData.value = data
    if (data.protein_pdb) proteinPdb.value = data.protein_pdb
  } catch (err) {
    runError.value = `Could not load pose: ${err.message}`
  } finally {
    poseLoading.value = false
  }
}

function onFilter(key) {
  leaderboardFilter.value = key
  loadLeaderboard()
}

/**
 * Re-open a finished run from the leaderboard.
 *
 * Nothing is recomputed: /api/run/<name> re-reads the stored outputs of a completed
 * Flyte run, so a visitor can come back hours later and still see their molecules and
 * poses. The entry that was clicked is pre-selected rather than just defaulting to
 * rank 1, because they clicked that specific molecule.
 */
async function openRun(entry) {
  if (busy.value) return
  busy.value = true
  runError.value = null
  candidates.value = []
  selectedIndex.value = null
  poseData.value = null

  try {
    const status = await api.runStatus(entry.run_name)
    if (status.error) {
      runError.value = status.error
      return
    }
    run.value = { ...status, run_name: entry.run_name, url: status.url || entry.run_url }
    foldLive.value = Boolean(status.folded_in_run)
    candidates.value = status.candidates ?? []

    const idx = candidates.value.findIndex((c) => c.smiles === entry.smiles)
    if (candidates.value.length) await selectCandidate(idx >= 0 ? idx : 0)

    if (status.target_key) {
      api.structure(status.target_key).then((s) => { proteinPdb.value = s.pdb }).catch(() => {})
    }
  } catch (err) {
    runError.value = `Could not open that run: ${err.message}`
  } finally {
    busy.value = false
  }
}

onMounted(async () => {
  await Promise.all([loadCatalogue(), loadLeaderboard(), loadNims()])
  timer = setInterval(() => {
    loadLeaderboard()
    loadNims()
  }, 10000)
})

onUnmounted(() => clearInterval(timer))
</script>

<template>
  <div class="shell">
    <header class="hero">
      <div class="hero-main">
        <p class="eyebrow">Union &middot; Flyte 2 &middot; NVIDIA BioNeMo</p>
        <h1>Generative Virtual Screening</h1>
        <div class="nim-badges">
          <span class="badge on">GENMOL</span>
          <span class="badge on">DIFFDOCK</span>
        </div>
        <p class="lede">
          NVIDIA's generative virtual screening blueprint, with the NIM microservices self-hosted on
          Union and the pipeline orchestrated by Flyte&nbsp;2. Generate small molecules against a
          target, dock them, and rank by pose confidence. Every click is a real workflow run.
        </p>
      </div>

      <div class="hero-side">
        <span class="pill" :class="nimsReady ? 'pill-ok' : 'pill-warn'">
          {{ nimsReady ? 'NIMs warm' : 'NIMs starting' }}
        </span>
        <div class="nim-list">
          <span v-for="n in boothNims" :key="n.name" class="nim" :title="n.detail">
            <i :class="n.status === 'ready' ? 'dot-ok' : 'dot-off'" />
            {{ n.name.replace('nim-', '') }}
          </span>
        </div>
        <button class="link-btn" @click="showLeaderboard = !showLeaderboard">
          {{ showLeaderboard ? 'Hide' : 'Show' }} leaderboard
        </button>
      </div>
    </header>

    <div v-if="bootError" class="panel boot-error">{{ bootError }}</div>

    <main v-else-if="catalogue" class="layout" :class="{ 'with-board': showLeaderboard }">
      <div class="col-input">
        <InputPanel :catalogue="catalogue" :busy="busy" @run="onRun" />
      </div>

      <div class="col-output">
        <OutputPanel
          :candidates="candidates"
          :selected-index="selectedIndex"
          :pose-data="poseData"
          :protein-pdb="proteinPdb"
          :pose-loading="poseLoading"
          :run="run"
          :busy="busy"
          @select="selectCandidate"
        />
        <!-- Also rendered when there is no run: if launching fails the run object is
             never created, and gating on `run` alone made a failed launch look like a
             dead button with no message anywhere. -->
        <RunProgress v-if="run || runError" :run="run" :fold-live="foldLive" :error="runError" />
      </div>

      <aside v-if="showLeaderboard" class="col-board">
        <Leaderboard
          :entries="leaderboard.entries"
          :stats="leaderboard.stats"
          :targets="catalogue.targets"
          :filter-key="leaderboardFilter"
          :highlight-run="run?.run_name"
          :selected-smiles="selectedIndex != null ? candidates[selectedIndex]?.smiles : null"
          @filter="onFilter"
          @open="openRun"
        />
      </aside>
    </main>

    <div v-else class="loading">Loading…</div>
  </div>
</template>

<style scoped>
.shell {
  max-width: 1860px;
  margin: 0 auto;
  padding: 22px 26px 56px;
}

.hero {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 28px;
  padding-bottom: 20px;
  margin-bottom: 20px;
  border-bottom: 1px solid var(--border);
}

.eyebrow {
  margin: 0 0 4px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
}

.hero h1 {
  font-size: 27px;
  margin-bottom: 10px;
}

.nim-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}

.badge {
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.07em;
  border-radius: 4px;
  padding: 3px 7px;
}

.badge.on {
  color: var(--accent);
  border: 1px solid var(--accent-dim);
}

.lede {
  margin: 0;
  color: var(--text-dim);
  font-size: 13.5px;
  max-width: 760px;
  line-height: 1.6;
}

.hero-side {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
  flex-shrink: 0;
}

.nim-list {
  display: flex;
  gap: 11px;
}

.nim {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--text-dim);
  font-family: var(--mono);
}

.dot-ok,
.dot-off {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  display: inline-block;
}

.dot-ok {
  background: var(--ok);
}

.dot-off {
  background: var(--border-bright);
}

.link-btn {
  background: none;
  color: var(--accent);
  font-size: 12px;
  font-weight: 600;
  padding: 0;
}

.layout {
  display: grid;
  grid-template-columns: 390px minmax(0, 1fr);
  gap: 22px;
  align-items: start;
}

.layout.with-board {
  grid-template-columns: 390px minmax(0, 1fr) 360px;
}

.col-input,
.col-output,
.col-board {
  min-width: 0;
}

.col-output {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.boot-error {
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-left: 3px solid var(--error);
  border-radius: var(--radius);
  padding: 18px;
  color: #ffbdbd;
}

.loading {
  color: var(--text-dim);
  padding: 56px 0;
  text-align: center;
}

@media (max-width: 1500px) {
  .layout,
  .layout.with-board {
    grid-template-columns: 360px minmax(0, 1fr);
  }
  .col-board {
    grid-column: 1 / -1;
  }
}

@media (max-width: 1000px) {
  .shell {
    padding: 16px 14px 40px;
  }
  .layout,
  .layout.with-board {
    grid-template-columns: 1fr;
  }
  .hero {
    flex-direction: column;
  }
  .hero-side {
    align-items: flex-start;
  }
}
</style>
