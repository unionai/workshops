<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue'

// 3Dmol is ~700 KB and isn't needed until a run produces a pose, so it is loaded on
// first render instead of being bundled into the initial payload.
let $3Dmol = null
async function load3Dmol() {
  $3Dmol ||= await import('3dmol')
  return $3Dmol
}

const props = defineProps({
  // PDB text for the receptor.
  proteinPdb: { type: String, default: null },
  // SDF / MOL-block text for the docked ligand pose.
  pose: { type: String, default: null },
  smiles: { type: String, default: null },
  confidence: { type: Number, default: null },
  loading: { type: Boolean, default: false },
})

const host = ref(null)
// Set when 3Dmol can't get a WebGL context: a locked-down browser, a machine with
// no GPU, or a remote session. Without this the failure is an unhandled exception
// inside a watcher and the whole panel goes blank with no explanation.
const webglError = ref(null)
let viewer = null

async function render() {
  if (!host.value || !props.proteinPdb) return

  try {
    const mol = await load3Dmol()

    // 3Dmol keeps internal GL state keyed to the container, so the viewer is
    // created once and only its models are swapped.
    if (!viewer) {
      viewer = mol.createViewer(host.value, { backgroundColor: '#0b0d10' })
    }

    viewer.removeAllModels()
    viewer.removeAllSurfaces()

    viewer.addModel(props.proteinPdb, 'pdb')
    viewer.setStyle({ model: 0 }, { cartoon: { color: 'spectrum', opacity: 0.72 } })

    if (props.pose) {
      viewer.addModel(props.pose, 'sdf')
      viewer.setStyle({ model: 1 }, { stick: { radius: 0.17, colorscheme: 'greenCarbon' } })
      // Frame on the ligand rather than the whole protein: the pocket is the
      // interesting part and a whole-protein zoom makes the ligand a dot.
      viewer.zoomTo({ model: 1 })
      viewer.zoom(0.45)
    } else {
      viewer.zoomTo()
    }

    viewer.render()
    webglError.value = null
  } catch (err) {
    // Most likely no WebGL context. Everything else on the page still works, so
    // degrade this panel rather than letting the error escape the watcher.
    viewer = null
    webglError.value = String(err?.message || err)
  }
}

function resetView() {
  if (!viewer) return
  try {
    viewer.zoomTo(props.pose ? { model: 1 } : {})
    if (props.pose) viewer.zoom(0.45)
    viewer.render()
  } catch {
    // A failed re-zoom is cosmetic; leave the current view alone.
  }
}

onMounted(render)
watch(() => [props.proteinPdb, props.pose], render)

onUnmounted(() => {
  // Release the WebGL context; a booth session may swap views many times.
  viewer?.clear()
  viewer = null
})
</script>

<template>
  <div class="panel viewer-panel">
    <div class="viewer-head">
      <div class="panel-title" style="margin: 0">Binding pose</div>
      <button v-if="pose" class="btn-ghost reset" @click="resetView">Reset view</button>
    </div>

    <div v-if="smiles" class="viewer-meta">
      <code>{{ smiles }}</code>
      <span v-if="confidence != null" class="conf">pose confidence {{ confidence.toFixed(3) }}</span>
    </div>

    <div ref="host" class="viewer" />

    <div v-if="webglError" class="overlay">
      <div class="webgl-msg">
        <strong>3D rendering unavailable</strong>
        <span>This browser could not create a WebGL context. Everything else still works.</span>
        <code>{{ webglError.slice(0, 120) }}</code>
      </div>
    </div>
    <div v-else-if="loading" class="overlay">Loading pose…</div>
    <div v-else-if="!proteinPdb" class="overlay">Run a screen to see a structure here.</div>
    <div v-else-if="!pose" class="overlay subtle">Select a candidate to see its pose.</div>
  </div>
</template>

<style scoped>
.viewer-panel {
  position: relative;
  display: flex;
  flex-direction: column;
}

.viewer-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}

.reset {
  padding: 5px 11px;
  font-size: 12px;
}

.viewer-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  font-size: 12px;
}

.viewer-meta code {
  color: var(--text-dim);
  word-break: break-all;
}

.conf {
  color: var(--accent);
  font-weight: 700;
  white-space: nowrap;
}

.viewer {
  position: relative;
  width: 100%;
  height: 380px;
  border-radius: 8px;
  overflow: hidden;
  background: var(--bg);
}

.overlay {
  position: absolute;
  left: 20px;
  right: 20px;
  bottom: 20px;
  height: 380px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-dim);
  font-size: 14px;
  pointer-events: none;
  background: rgba(11, 13, 16, 0.72);
  border-radius: 8px;
}

.overlay.subtle {
  background: transparent;
}

.webgl-msg {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 7px;
  text-align: center;
  padding: 0 28px;
  max-width: 460px;
}

.webgl-msg strong {
  color: var(--warn);
  font-size: 15px;
}

.webgl-msg code {
  font-size: 11px;
  color: var(--text-dim);
  word-break: break-all;
}
</style>
