/**
 * 浏览器本地保存的大模型配置。
 * active=local：未填项沿用服务器 .env；active=online：三项都填齐才改用在线模型。
 */

const STORAGE_KEY = 'litcraft.llmConfig'
export const LLM_CONFIG_EVENT = 'litcraft-llm-config'

export function emptyProfile() {
  return { model: '', apiKey: '', baseUrl: '' }
}

export function emptyLlmStore() {
  return {
    active: 'local',
    local: emptyProfile(),
    online: emptyProfile(),
  }
}

function trimProfile(profile) {
  return {
    model: String(profile?.model || '').trim(),
    apiKey: String(profile?.apiKey || '').trim(),
    baseUrl: String(profile?.baseUrl || '').trim(),
  }
}

function hasAnyField(profile) {
  const current = trimProfile(profile)
  return Boolean(current.model || current.apiKey || current.baseUrl)
}

export function isCompleteProfile(profile) {
  const current = trimProfile(profile)
  return Boolean(current.model && current.apiKey && current.baseUrl)
}

function migrateLegacy(parsed) {
  const store = emptyLlmStore()
  const preset = parsed.preset || 'local'
  const profile = {
    model: String(parsed.model || '').trim(),
    apiKey: String(parsed.apiKey || '').trim(),
    baseUrl: String(parsed.baseUrl || '').trim(),
  }
  if (preset === 'local') {
    store.local = profile
    store.active = 'local'
  } else {
    store.online = profile
    store.active = isCompleteProfile(profile) ? 'online' : 'local'
  }
  return store
}

export function loadLlmConfig() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return emptyLlmStore()
    const parsed = JSON.parse(raw)
    if (parsed.local || parsed.online || parsed.active) {
      return {
        active: parsed.active === 'online' ? 'online' : 'local',
        local: { ...emptyProfile(), ...trimProfile(parsed.local) },
        online: { ...emptyProfile(), ...trimProfile(parsed.online) },
      }
    }
    return migrateLegacy(parsed)
  } catch {
    return emptyLlmStore()
  }
}

export function saveLlmConfig(store) {
  const next = {
    active: store.active === 'online' ? 'online' : 'local',
    local: trimProfile(store.local),
    online: trimProfile(store.online),
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  window.dispatchEvent(new Event(LLM_CONFIG_EVENT))
  return next
}

export function llmPayloadForStart(store = loadLlmConfig()) {
  const current = store.active === 'online' ? store.online : store.local
  if (store.active === 'online') {
    if (!isCompleteProfile(current)) return {}
    const online = trimProfile(current)
    return {
      llm_source: 'online',
      llm_model_id: online.model,
      llm_api_key: online.apiKey,
      llm_base_url: online.baseUrl,
    }
  }
  if (!hasAnyField(store.local)) return {}
  const local = trimProfile(store.local)
  const payload = { llm_source: 'local' }
  if (local.model) payload.llm_model_id = local.model
  if (local.apiKey) payload.llm_api_key = local.apiKey
  if (local.baseUrl) payload.llm_base_url = local.baseUrl
  return payload
}

export function llmStatusLabel(store = loadLlmConfig()) {
  if (store.active === 'online' && isCompleteProfile(store.online)) {
    return String(store.online.model || '').trim() || '在线'
  }
  const localModel = String(store.local?.model || '').trim()
  return localModel || '本地'
}
