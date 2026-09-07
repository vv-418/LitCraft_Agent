<template>
  <div>
    <h1 class="page-title">历史记录</h1>
    <p class="page-desc">查看以往任务摘要，点击可打开对应综述结果。</p>

    <div v-if="loading" class="empty">加载中…</div>
    <div v-else-if="error" class="alert error">{{ error }}</div>
    <div v-else-if="!tasks.length" class="empty">暂无历史任务记录</div>

    <div v-else class="history-list">
      <article v-for="task in tasks" :key="task.task_id" class="history-item">
        <h3>{{ task.topic }}</h3>
        <div class="history-meta">
          <span class="badge" :class="task.status">{{ statusLabel(task.status) }}</span>
          <span>创建时间：{{ task.created_at }}</span>
          <span>ID：{{ task.task_id.slice(0, 8) }}…</span>
        </div>
        <div>
          <button class="btn btn-link" type="button" @click="viewTask(task)">
            查看结果
          </button>
        </div>
      </article>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { getHistory } from '../api/client'

const router = useRouter()
const tasks = ref([])
const loading = ref(true)
const error = ref('')

function statusLabel(status) {
  if (status === 'done') return '完成'
  if (status === 'error') return '错误'
  return '运行中'
}

function viewTask(task) {
  router.push({ path: '/', query: { task_id: task.task_id } })
}

onMounted(async () => {
  loading.value = true
  error.value = ''
  try {
    tasks.value = await getHistory()
  } catch (e) {
    console.error(e)
    error.value = '无法连接后端服务'
  } finally {
    loading.value = false
  }
})
</script>
