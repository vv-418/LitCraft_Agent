<template>
  <div class="panel">
    <h2 style="margin: 0 0 12px; font-size: 1.2rem">文献综述正文</h2>

    <div v-if="html" class="review-body" v-html="html"></div>
    <p v-else class="empty">最终答案为空</p>

    <p v-if="papersFolder" class="empty" style="padding-top: 0; margin-bottom: 12px">
      下载论文：<code>{{ papersFolder }}</code>
    </p>
    <p v-if="savedPath" class="empty" style="padding-top: 0; margin-bottom: 12px">
      综述已保存到：<code>{{ savedPath }}</code>
    </p>

    <div v-if="canSave" style="margin-top: 20px; display: flex; gap: 10px; flex-wrap: wrap">
      <button
        class="btn btn-primary"
        type="button"
        :disabled="saving"
        style="width: auto"
        @click="onSave"
      >
        {{ saving ? '保存中…' : '保存 PDF' }}
      </button>
    </div>
    <div v-if="saveError" class="alert error" style="margin-top: 12px">{{ saveError }}</div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import MarkdownIt from 'markdown-it'
import { pickSavePath, saveReviewPdf } from '../api/client'

const props = defineProps({
  finalAnswer: { type: String, default: '' },
  pdfPath: { type: String, default: '' },
  taskId: { type: String, default: '' },
  topic: { type: String, default: '' },
  papersFolder: { type: String, default: '' },
  reviewFolder: { type: String, default: '' },
})

const md = new MarkdownIt({ html: false, linkify: true, breaks: false })
const saving = ref(false)
const saveError = ref('')
const savedPath = ref('')

const html = computed(() => (props.finalAnswer ? md.render(props.finalAnswer) : ''))
const canSave = computed(() => Boolean(props.taskId && props.finalAnswer))

watch(
  () => [props.pdfPath, props.reviewFolder],
  () => {
    if (props.pdfPath) savedPath.value = props.pdfPath
    else if (props.reviewFolder) savedPath.value = props.reviewFolder
  },
  { immediate: true },
)

async function onSave() {
  if (!canSave.value || saving.value) return
  saveError.value = ''
  saving.value = true
  try {
    const picked = await pickSavePath({
      kind: 'review',
      topic: props.topic,
    })
    if (!picked?.directory) {
      return
    }
    const saved = await saveReviewPdf(props.taskId, {
      review_output_dir: picked.directory,
      review_filename: picked.filename || '',
    })
    savedPath.value = saved.pdf_path || picked.directory
  } catch (e) {
    saveError.value = '保存失败，请确认已选择位置且后端正在运行'
    console.error(e)
  } finally {
    saving.value = false
  }
}
</script>
