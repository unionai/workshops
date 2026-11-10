<script setup>
import { onMounted, ref, watch } from 'vue'
import SmilesDrawer from 'smiles-drawer'

const props = defineProps({
  smiles: { type: String, required: true },
  width: { type: Number, default: 220 },
  height: { type: Number, default: 160 },
})

const canvas = ref(null)
const failed = ref(false)

// One drawer instance per component is fine; it holds only theme/size options.
let drawer = null

function draw() {
  if (!canvas.value) return

  drawer ||= new SmilesDrawer.Drawer({
    width: props.width,
    height: props.height,
    bondThickness: 1.1,
    bondLength: 16,
    atomVisualization: 'default',
    // Light-on-dark palette to match the booth theme. Without this, the default
    // dark-on-white atoms are invisible against the panel.
    themes: {
      booth: {
        C: '#eef1f5',
        O: '#ff6b6b',
        N: '#6ba8ff',
        S: '#ffd93d',
        F: '#3ddc97',
        CL: '#3ddc97',
        BR: '#d98c3d',
        I: '#c77dff',
        P: '#ffa94d',
        BACKGROUND: '#14181d',
      },
    },
  })

  // SmilesDrawer.parse is forgiving but not infallible; GenMol can emit a string
  // that won't parse. Show a fallback rather than an empty box.
  SmilesDrawer.parse(
    props.smiles,
    (tree) => {
      failed.value = false
      drawer.draw(tree, canvas.value, 'booth', false)
    },
    () => {
      failed.value = true
    },
  )
}

onMounted(draw)
watch(() => props.smiles, draw)
</script>

<template>
  <div class="mol" :style="{ width: `${width}px`, height: `${height}px` }">
    <canvas v-show="!failed" ref="canvas" :width="width" :height="height" />
    <div v-if="failed" class="mol-failed">
      <span>Could not render</span>
      <code>{{ smiles.slice(0, 40) }}{{ smiles.length > 40 ? '…' : '' }}</code>
    </div>
  </div>
</template>

<style scoped>
.mol {
  background: var(--bg-raised);
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.mol-failed {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  color: var(--text-dim);
  font-size: 11px;
  text-align: center;
  padding: 10px;
  word-break: break-all;
}
</style>
