<template>
  <form class="panel form-grid" @submit.prevent="onSubmit">
    <div class="field">
      <div class="field-head">
        研究主题
        <HelpTip>要综述的学术主题，支持中英文。</HelpTip>
      </div>
      <input
        v-model="topic"
        type="text"
        placeholder="例：Transformer 在自然语言处理中的应用"
        required
      />
    </div>

    <div class="field">
      <div class="field-head">
        起始年份
        <HelpTip>只搜该年及之后的文献；填 0 表示不限。</HelpTip>
      </div>
      <input v-model.number="year" type="number" min="0" max="2100" />
    </div>

    <div class="field-row">
      <div class="field">
        <div class="field-head">
          单源保留篇数
          <HelpTip>每个来源（arXiv / S2 / Scholar）每次最多取多少篇。</HelpTip>
        </div>
        <input v-model.number="perSourceLimit" type="number" min="1" max="50" />
      </div>
      <div class="field">
        <div class="field-head">
          最终保留篇数
          <HelpTip>多源合并去重后对外保留多少篇。</HelpTip>
        </div>
        <input v-model.number="finalLimit" type="number" min="1" max="50" />
      </div>
    </div>

    <div class="field">
      <div class="field-head">
        保存位置
        <HelpTip>论文默认写入 output/日期/主题/lit_source/，结束后按「标题__网址」命名。点浏览可改文件夹。</HelpTip>
      </div>
      <div class="path-row">
        <input
          v-model="papersSavePath"
          type="text"
          placeholder="默认：项目 output/日期/主题/lit_source/"
        />
        <button class="btn btn-secondary" type="button" :disabled="pickingPapers" @click="onPickSave">
          {{ pickingPapers ? '选择中…' : '浏览' }}
        </button>
      </div>
    </div>

    <button class="btn btn-primary" type="submit" :disabled="submitting || !topic.trim()">
      {{ submitting ? '提交中…' : '开始生成综述' }}
    </button>

    <div v-if="error" class="alert error">{{ error }}</div>
  </form>
</template>

<script setup>
import { ref } from 'vue'
import { startTask, pickSavePath } from '../api/client'
import { llmPayloadForStart } from '../llmConfig'
import HelpTip from './HelpTip.vue'

const emit = defineEmits(['started'])

const topic = ref('')
const year = ref(0)
const perSourceLimit = ref(10)
const finalLimit = ref(10)
const papersSavePath = ref('')
const pickingPapers = ref(false)
const submitting = ref(false)
const error = ref('')

function clampLimit(n, fallback = 10) {
  const v = Number(n)
  if (!Number.isFinite(v)) return fallback
  return Math.min(50, Math.max(1, Math.round(v)))
}

function splitSavePath(raw) {
  const text = (raw || '').trim().replace(/^["']|["']$/g, '')
  if (!text) return { directory: '', filename: '' }
  const isFile = /\.[a-z0-9]{2,8}$/i.test(text)
  if (!isFile) return { directory: text, filename: '' }
  const lastSlash = Math.max(text.lastIndexOf('\\'), text.lastIndexOf('/'))
  if (lastSlash < 0) return { directory: '', filename: text }
  return {
    directory: text.slice(0, lastSlash),
    filename: text.slice(lastSlash + 1),
  }
}

function papersPrefix(filename) {
  const name = (filename || '').trim()
  if (!name) return ''
  const stem = name.replace(/\.pdf$/i, '')
  if (['标题__网址', 'untitled', '未命名'].includes(stem)) return ''
  return stem
}

async function onPickSave() {
  pickingPapers.value = true
  error.value = ''
  try {
    const data = await pickSavePath({
      kind: 'papers',
      topic: topic.value.trim(),
    })
    if (data?.directory) {
      papersSavePath.value = data.filename
        ? `${data.directory}\\${data.filename}`.replace(/\//g, '\\')
        : data.directory
    }
  } catch (e) {
    error.value = '无法打开保存对话框，请直接粘贴文件夹路径'
    console.error(e)
  } finally {
    pickingPapers.value = false
  }
}

async function onSubmit() {
  error.value = ''
  const trimmed = topic.value.trim()
  if (!trimmed) {
    error.value = '请输入研究主题'
    return
  }

  submitting.value = true
  try {
    const papers = splitSavePath(papersSavePath.value)
    const data = await startTask({
      topic: trimmed,
      year_from: year.value > 0 ? String(year.value) : '',
      save_pdf: false,
      per_source_limit: clampLimit(perSourceLimit.value),
      final_limit: clampLimit(finalLimit.value),
      papers_output_dir: papers.directory,
      papers_filename: papersPrefix(papers.filename),
      ...llmPayloadForStart(),
    })
    emit('started', { taskId: data.task_id, topic: trimmed })
  } catch (e) {
    error.value = '无法连接后端服务，请确认 API 已启动（端口 8000）'
    console.error(e)
  } finally {
    submitting.value = false
  }
}
</script>
