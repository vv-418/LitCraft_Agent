<template>
  <div class="help-tip" ref="root">
    <button
      class="help-btn"
      type="button"
      aria-label="说明"
      :class="{ open }"
      @click.stop="toggle"
    >
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5" />
        <path
          d="M9.2 9.2a2.8 2.8 0 0 1 5.5.9c0 1.85-2.75 2.35-2.75 4"
          stroke="currentColor"
          stroke-width="1.5"
          stroke-linecap="round"
        />
        <circle cx="12" cy="17.15" r="1" fill="currentColor" />
      </svg>
    </button>
    <div v-if="open" class="help-pop" role="dialog" @click.stop>
      <slot />
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue'

const HELP_EVENT = 'litcraft-help-toggle'
const id = `help-${Math.random().toString(36).slice(2, 8)}`
const open = ref(false)
const root = ref(null)

function toggle() {
  if (open.value) {
    open.value = false
    return
  }
  window.dispatchEvent(new CustomEvent(HELP_EVENT, { detail: id }))
  open.value = true
}

function onHelpEvent(event) {
  if (event.detail !== id) open.value = false
}

function onDocClick(event) {
  if (!open.value) return
  if (root.value && !root.value.contains(event.target)) {
    open.value = false
  }
}

onMounted(() => {
  window.addEventListener(HELP_EVENT, onHelpEvent)
  document.addEventListener('click', onDocClick)
})

onUnmounted(() => {
  window.removeEventListener(HELP_EVENT, onHelpEvent)
  document.removeEventListener('click', onDocClick)
})
</script>
