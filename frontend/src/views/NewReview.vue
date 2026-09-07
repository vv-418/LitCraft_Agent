<template>
  <div>
    <h1 class="page-title">新建文献综述</h1>
    <p class="page-desc">
      输入研究主题，LitCraft Agent 将自动搜索论文、下载 PDF、解析全文并生成综述。
    </p>

    <TaskForm v-if="!taskId" @started="onStarted" />

    <div v-if="taskId" class="panel" style="margin-top: 24px">
      <div class="meta-row">
        <span>任务 ID：<code>{{ shortId }}</code></span>
        <span>主题：{{ topic }}</span>
        <span class="badge" :class="status">{{ statusLabel }}</span>
      </div>

      <div v-if="status === 'error'" class="alert error">
        <div><strong>错误信息：</strong>{{ errorMessage || '未知错误' }}</div>
        <details v-if="traceback" style="margin-top: 10px">
          <summary style="cursor: pointer">查看完整错误堆栈</summary>
          <pre style="white-space: pre-wrap; margin-top: 8px">{{ traceback }}</pre>
        </details>
      </div>

      <StepTimeline :steps="steps" :max-steps="MAX_STEPS" :progress="progress" />

      <p v-if="status === 'running'" class="empty" style="padding-top: 8px">
        正在自动刷新获取最新进度。检索结束后会再写一篇完整综述，本地 14B 成稿可能需要几分钟。
      </p>

      <div v-if="status === 'done' || status === 'error'" style="margin-top: 18px">
        <button class="btn btn-secondary" type="button" @click="resetTask">
          生成新的综述
        </button>
      </div>
    </div>

    <div v-if="status === 'done' && finalAnswer !== null" style="margin-top: 24px">
      <ReviewResult
        :final-answer="finalAnswer"
        :pdf-path="pdfPath"
        :task-id="taskId"
        :topic="topic"
        :papers-folder="papersFolder"
        :review-folder="reviewFolder"
      />
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import TaskForm from '../components/TaskForm.vue'
import StepTimeline from '../components/StepTimeline.vue'
import ReviewResult from '../components/ReviewResult.vue'
import { getResult, getStatus, getSteps } from '../api/client'

const MAX_STEPS = 20
const POLL_MS = 2000

const route = useRoute()
const router = useRouter()

const taskId = ref('')
const topic = ref('')
const status = ref('')
const steps = ref([])
const finalAnswer = ref(null)
const pdfPath = ref('')
const outputFolder = ref('')
const papersFolder = ref('')
const reviewFolder = ref('')
const errorMessage = ref('')
const traceback = ref('')

let pollTimer = null

const shortId = computed(() => (taskId.value ? `${taskId.value.slice(0, 8)}…` : ''))
const progress = computed(() => {
  const done = steps.value.filter((s) => !s.pending).length
  return Math.min(done / MAX_STEPS, 1)
})
const statusLabel = computed(() => {
  if (status.value === 'running') return '执行中'
  if (status.value === 'done') return '完成'
  if (status.value === 'error') return '出错'
  return status.value || '未知'
})

function clearPoll() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function pollOnce() {
  if (!taskId.value) return
  try {
    const [statusData, stepsData] = await Promise.all([
      getStatus(taskId.value),
      getSteps(taskId.value),
    ])
    status.value = statusData.status
    steps.value = stepsData.steps || []

    if (status.value === 'done' || status.value === 'error') {
      clearPoll()
      const result = await getResult(taskId.value)
      topic.value = result.topic || topic.value
      finalAnswer.value = result.final_answer || ''
      pdfPath.value = result.pdf_path || ''
      outputFolder.value = result.output_folder || ''
      papersFolder.value = result.papers_folder || ''
      reviewFolder.value = result.review_folder || ''
      errorMessage.value = result.error || ''
      traceback.value = result.traceback || ''
      if (result.steps?.length) {
        steps.value = result.steps
      }
    }
  } catch (e) {
    console.error(e)
    clearPoll()
    status.value = 'error'
    errorMessage.value = '无法连接后端服务，请确认 API 服务已启动'
  }
}

function startPolling() {
  clearPoll()
  pollOnce()
  pollTimer = setInterval(pollOnce, POLL_MS)
}

function onStarted({ taskId: id, topic: t }) {
  taskId.value = id
  topic.value = t
  status.value = 'running'
  steps.value = []
  finalAnswer.value = null
  pdfPath.value = ''
  outputFolder.value = ''
  papersFolder.value = ''
  reviewFolder.value = ''
  errorMessage.value = ''
  traceback.value = ''
  router.replace({ query: { task_id: id } })
  startPolling()
}

function resetTask() {
  clearPoll()
  taskId.value = ''
  topic.value = ''
  status.value = ''
  steps.value = []
  finalAnswer.value = null
  pdfPath.value = ''
  outputFolder.value = ''
  papersFolder.value = ''
  reviewFolder.value = ''
  errorMessage.value = ''
  traceback.value = ''
  router.replace({ query: {} })
}

async function loadExistingTask(id, shouldPoll) {
  taskId.value = id
  status.value = 'running'
  steps.value = []
  finalAnswer.value = null
  try {
    const result = await getResult(id)
    topic.value = result.topic || ''
    status.value = result.status
    steps.value = result.steps || []
    if (result.status === 'done') {
      finalAnswer.value = result.final_answer || ''
      pdfPath.value = result.pdf_path || ''
      outputFolder.value = result.output_folder || ''
      papersFolder.value = result.papers_folder || ''
      reviewFolder.value = result.review_folder || ''
    } else if (result.status === 'error') {
      errorMessage.value = result.error || ''
      traceback.value = result.traceback || ''
    } else if (shouldPoll) {
      startPolling()
    }
  } catch (e) {
    console.error(e)
    status.value = 'error'
    errorMessage.value = '任务不存在或已被清除'
  }
}

watch(
  () => route.query.task_id,
  (id) => {
    if (id && id !== taskId.value) {
      clearPoll()
      loadExistingTask(String(id), true)
    }
  },
)

onMounted(() => {
  const id = route.query.task_id
  if (id) {
    loadExistingTask(String(id), true)
  }
})

onUnmounted(() => {
  clearPoll()
})
</script>
