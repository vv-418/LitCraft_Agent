<template>
  <div>
    <div class="progress" aria-hidden="true">
      <span :style="{ width: `${progress * 100}%` }"></span>
    </div>
    <div class="progress-label">
      执行进度：{{ completedCount }} / {{ maxSteps }} 步
      <span v-if="hasPending" class="progress-live">· 进行中</span>
    </div>

    <h2 style="margin: 0 0 14px; font-size: 1.15rem">执行轨迹</h2>

    <p v-if="!steps.length" class="empty">Agent 正在初始化……等待第一步输出</p>

    <div class="step-list">
      <article
        v-for="(step, index) in steps"
        :key="index"
        class="step-card"
        :class="{ pending: step.pending, error: !!summaryOf(step).error }"
      >
        <header class="step-head">
          <div class="step-head-main">
            <span class="step-index">步骤 {{ index + 1 }}</span>
            <span class="step-kind" :data-kind="summaryOf(step).kind">
              {{ summaryOf(step).title || step.action || '思考' }}
            </span>
          </div>
          <span v-if="step.pending" class="step-status pending">执行中</span>
          <span v-else-if="summaryOf(step).error" class="step-status error">失败</span>
          <span v-else class="step-status ok">完成</span>
        </header>

        <p class="step-headline">{{ summaryOf(step).headline || fallbackHeadline(step) }}</p>

        <ul v-if="summaryOf(step).papers?.length" class="paper-list">
          <li v-for="(paper, pi) in summaryOf(step).papers.slice(0, 12)" :key="pi">
            <div class="paper-title">{{ paper.title || '（无标题）' }}</div>
            <div class="paper-meta">
              <span v-if="paper.year">{{ paper.year }}</span>
              <span v-if="paper.source">{{ paper.source }}</span>
              <a
                v-if="paper.url"
                :href="paper.url"
                target="_blank"
                rel="noopener noreferrer"
              >打开</a>
            </div>
          </li>
        </ul>
        <p
          v-if="(summaryOf(step).papers?.length || 0) > 12"
          class="paper-more"
        >
          另有 {{ summaryOf(step).papers.length - 12 }} 篇未全部展开，见下方原始输出
        </p>

        <details class="raw-block">
          <summary>查看原始输入 / 输出</summary>
          <div class="step-body">
            <div v-if="step.thought">
              <strong>思考：</strong>{{ step.thought }}
            </div>
            <div v-if="step.action">
              <strong>行动：</strong><code>{{ step.action }}</code>
            </div>
            <div v-if="step.action_input && Object.keys(step.action_input).length">
              <strong>输入：</strong>
              <pre>{{ formatInput(step.action_input) }}</pre>
            </div>
            <div v-if="step.observation">
              <strong>观察：</strong>
              <pre>{{ truncate(step.observation) }}</pre>
            </div>
          </div>
        </details>
      </article>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  steps: { type: Array, default: () => [] },
  maxSteps: { type: Number, default: 20 },
  progress: { type: Number, default: 0 },
})

const hasPending = computed(() => props.steps.some((s) => s.pending))
const completedCount = computed(
  () => props.steps.filter((s) => !s.pending).length,
)

function summaryOf(step) {
  return step?.summary || {}
}

function fallbackHeadline(step) {
  if (step.pending) return '正在执行…'
  if (step.action) return `已调用 ${step.action}`
  if (step.thought) return String(step.thought).slice(0, 120)
  return '已完成一步'
}

function formatInput(input) {
  try {
    return JSON.stringify(input, null, 2)
  } catch {
    return String(input)
  }
}

function truncate(text) {
  if (!text) return ''
  return text.length > 4000 ? `${text.slice(0, 4000)}...` : text
}
</script>
