<template>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">
        <h1>LitCraft</h1>
        <p>智能文献综述生成助手</p>
      </div>

      <nav class="nav">
        <router-link to="/">新建综述</router-link>
        <router-link to="/history">历史记录</router-link>
        <router-link to="/settings">模型配置</router-link>
      </nav>

      <div class="sidebar-llm">
        当前模型：{{ llmLabel }}
      </div>

      <div class="sidebar-footer">
        <span class="status-dot" :class="apiOk ? 'ok' : 'bad'"></span>
        后端状态：{{ apiOk ? '运行中' : '未连接' }}
      </div>
    </aside>

    <main class="main">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { checkHealth } from './api/client'
import { LLM_CONFIG_EVENT, llmStatusLabel, loadLlmConfig } from './llmConfig'

const apiOk = ref(false)
const llmLabel = ref(llmStatusLabel())
let timer = null

function refreshLlmLabel() {
  llmLabel.value = llmStatusLabel(loadLlmConfig())
}

async function refreshHealth() {
  apiOk.value = await checkHealth()
}

onMounted(() => {
  refreshHealth()
  refreshLlmLabel()
  timer = setInterval(refreshHealth, 10000)
  window.addEventListener(LLM_CONFIG_EVENT, refreshLlmLabel)
  window.addEventListener('storage', refreshLlmLabel)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
  window.removeEventListener(LLM_CONFIG_EVENT, refreshLlmLabel)
  window.removeEventListener('storage', refreshLlmLabel)
})
</script>
