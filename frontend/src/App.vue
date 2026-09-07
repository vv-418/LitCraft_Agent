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
      </nav>

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

const apiOk = ref(false)
let timer = null

async function refreshHealth() {
  apiOk.value = await checkHealth()
}

onMounted(() => {
  refreshHealth()
  timer = setInterval(refreshHealth, 10000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>
