/**
 * LitCraft API 客户端。
 * 业务接口走 Vite proxy（相对路径 /api），健康检查直连后端根路径。
 */

const API_ORIGIN = import.meta.env.VITE_API_ORIGIN || 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function checkHealth() {
  try {
    const res = await fetch(`${API_ORIGIN}/`, { method: 'GET' })
    return res.ok
  } catch {
    return false
  }
}

export function startTask({
  topic,
  year_from = '',
  save_pdf = false,
  per_source_limit = 10,
  final_limit = 10,
  output_dir = '',
  papers_output_dir = '',
  papers_filename = '',
  review_output_dir = '',
  review_filename = '',
  llm_model_id = '',
  llm_api_key = '',
  llm_base_url = '',
  llm_source = '',
}) {
  const body = {
    topic,
    year_from,
    save_pdf,
    per_source_limit,
    final_limit,
    output_dir,
    papers_output_dir,
    papers_filename,
    review_output_dir,
    review_filename,
  }
  if (llm_source === 'local') {
    body.llm_source = 'local'
    if (llm_model_id) body.llm_model_id = llm_model_id
    if (llm_api_key) body.llm_api_key = llm_api_key
    if (llm_base_url) body.llm_base_url = llm_base_url
  } else if (llm_source === 'online' && llm_model_id && llm_api_key && llm_base_url) {
    body.llm_source = 'online'
    body.llm_model_id = llm_model_id
    body.llm_api_key = llm_api_key
    body.llm_base_url = llm_base_url
  }
  return request('/api/agent/start', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function getStatus(taskId) {
  return request(`/api/agent/status/${taskId}`)
}

export function getSteps(taskId) {
  return request(`/api/agent/steps/${taskId}`)
}

export function getResult(taskId) {
  return request(`/api/agent/result/${taskId}`)
}

export function getHistory() {
  return request('/api/history')
}

export function pdfUrl(filename) {
  return `/api/output/${encodeURIComponent(filename)}`
}

export function taskPdfUrl(taskId) {
  return `/api/agent/pdf/${encodeURIComponent(taskId)}`
}

export function pickDirectory() {
  return request('/api/pick-directory', { method: 'POST' })
}

export function pickSavePath({ kind = 'review', topic = '', initial_file = '' } = {}) {
  return request('/api/pick-save-path', {
    method: 'POST',
    body: JSON.stringify({ kind, topic, initial_file }),
  })
}

export function saveReviewPdf(taskId, { review_output_dir, review_filename = '' }) {
  return request(`/api/agent/save-pdf/${encodeURIComponent(taskId)}`, {
    method: 'POST',
    body: JSON.stringify({ review_output_dir, review_filename }),
  })
}

export function getLlmInfo() {
  return request('/api/llm/info')
}
