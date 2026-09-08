<template>
  <div>
    <h1 class="page-title">模型配置</h1>

    <div class="settings-tabs">
      <button
        class="settings-tab"
        :class="{ active: tab === 'local', inuse: store.active === 'local' }"
        type="button"
        @click="tab = 'local'"
      >
        本地模型配置
      </button>
      <button
        class="settings-tab"
        :class="{ active: tab === 'online', inuse: store.active === 'online' }"
        type="button"
        @click="tab = 'online'"
      >
        在线模型配置
      </button>
    </div>

    <form class="panel form-grid" @submit.prevent="onSave">
      <div class="field">
        <div class="field-head">
          模型 ID
          <HelpTip>{{ tab === 'local' ? localTips.model : onlineTips.model }}</HelpTip>
        </div>
        <input
          v-model="profile.model"
          type="text"
          :placeholder="placeholders.model"
          autocomplete="off"
        />
      </div>

      <div class="field">
        <div class="field-head">
          接口地址
          <HelpTip>{{ tab === 'local' ? localTips.baseUrl : onlineTips.baseUrl }}</HelpTip>
        </div>
        <input
          v-model="profile.baseUrl"
          type="text"
          :placeholder="placeholders.baseUrl"
          autocomplete="off"
        />
      </div>

      <div class="field">
        <div class="field-head">
          API Key
          <HelpTip>{{ tab === 'local' ? localTips.apiKey : onlineTips.apiKey }}</HelpTip>
        </div>
        <div class="path-row">
          <input
            v-model="profile.apiKey"
            :type="showKey ? 'text' : 'password'"
            :placeholder="placeholders.apiKey"
            autocomplete="off"
          />
          <button class="btn btn-secondary" type="button" @click="showKey = !showKey">
            {{ showKey ? '隐藏' : '显示' }}
          </button>
        </div>
      </div>

      <div v-if="savedMessage" class="alert ok">{{ savedMessage }}</div>
      <div v-if="error" class="alert error">{{ error }}</div>

      <div class="form-actions">
        <button class="btn btn-primary" type="submit">保存并使用</button>
        <button class="btn btn-secondary" type="button" @click="onClear">清空</button>
      </div>
    </form>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import HelpTip from '../components/HelpTip.vue'
import { getLlmInfo } from '../api/client'
import {
  emptyProfile,
  isCompleteProfile,
  loadLlmConfig,
  saveLlmConfig,
} from '../llmConfig'

const localTips = {
  model: '本机模型名，例如 qwen2.5:14b-instruct。留空沿用服务器 .env。',
  baseUrl: '本机 OpenAI 兼容地址。Ollama 一般为 http://localhost:11434/v1。留空沿用 .env。',
  apiKey: '本地常用 ollama 或任意非空值。留空沿用 .env。密钥只存在本机浏览器。',
}

const onlineTips = {
  model: '在线模型名。DeepSeek 写综述用 deepseek-chat。',
  baseUrl: '服务商文档中的 Base URL。DeepSeek 为 https://api.deepseek.com/v1。',
  apiKey: '开放平台创建的密钥，一般以 sk- 开头。三项都填齐并保存后才会改用在线模型，否则仍用本地。密钥只存在本机浏览器。',
}

const store = reactive(loadLlmConfig())
const tab = ref(store.active === 'online' ? 'online' : 'local')
const defaults = reactive({ model: '', base_url: '' })
const showKey = ref(false)
const savedMessage = ref('')
const error = ref('')

const profile = computed(() => (tab.value === 'online' ? store.online : store.local))

const placeholders = computed(() => {
  if (tab.value === 'online') {
    return {
      model: 'deepseek-chat',
      baseUrl: 'https://api.deepseek.com/v1',
      apiKey: 'sk-…',
    }
  }
  return {
    model: defaults.model || '本地模型 ID',
    baseUrl: defaults.base_url || 'http://127.0.0.1:11434/v1',
    apiKey: '本地服务密钥',
  }
})

watch(tab, () => {
  savedMessage.value = ''
  error.value = ''
})

function onSave() {
  error.value = ''
  savedMessage.value = ''
  if (tab.value === 'online' && !isCompleteProfile(store.online)) {
    error.value = '请填齐三项后再使用在线模型'
    return
  }
  store.active = tab.value
  saveLlmConfig(store)
  savedMessage.value = '已保存'
}

function onClear() {
  error.value = ''
  savedMessage.value = ''
  if (tab.value === 'online') {
    store.online = emptyProfile()
    if (store.active === 'online') store.active = 'local'
  } else {
    store.local = emptyProfile()
    store.active = 'local'
  }
  saveLlmConfig(store)
  savedMessage.value = '已清空'
}

onMounted(async () => {
  try {
    const info = await getLlmInfo()
    defaults.model = info.model || ''
    defaults.base_url = info.base_url || ''
  } catch {
    /* 占位仍可用 */
  }
})
</script>
