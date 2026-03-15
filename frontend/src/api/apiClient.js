/**
 * apiClient.js
 * Centralised Axios instance + fetch-based streaming for all backend communication.
 *
 * Exports:
 *   uploadDocument(file, collection, onUploadProgress) → POST /api/v1/upload
 *   queryDocuments(question, topK, collection, conversationId) → POST /api/v1/query
 *   streamQuery(question, onToken, onDone, topK, collection, conversationId)
 *   listDocuments(collection) → GET /api/v1/documents
 *   listCollections()        → GET /api/v1/collections
 *   startConversation()      → POST /api/v1/conversations/start
 *   evaluateRag()            → GET /api/v1/evaluate
 */

import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 180_000, // 3-minute timeout — LLM generation can be slow
  headers: { Accept: 'application/json' },
})

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred.'
    error.userMessage = detail
    return Promise.reject(error)
  },
)

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

/**
 * Upload a single document to the backend for indexing.
 * @param {File}     file
 * @param {string}   collection
 * @param {Function} onUploadProgress
 */
export async function uploadDocument(file, collection = 'default', onUploadProgress) {
  const form = new FormData()
  form.append('file', file)

  const response = await apiClient.post(
    `/api/v1/upload?collection=${encodeURIComponent(collection)}`,
    form,
    { headers: { 'Content-Type': 'multipart/form-data' }, onUploadProgress },
  )
  return response.data
}

/**
 * Send a question to the RAG pipeline and receive a full answer.
 * @param {string}        question
 * @param {number}        topK
 * @param {string}        collection
 * @param {string|null}   conversationId
 */
export async function queryDocuments(
  question,
  topK = 5,
  collection = 'default',
  conversationId = null,
) {
  const response = await apiClient.post('/api/v1/query', {
    question,
    top_k: topK,
    collection,
    conversation_id: conversationId,
  })
  return response.data
}

/**
 * Streaming query using fetch + ReadableStream (SSE).
 *
 * @param {string}        question
 * @param {Function}      onToken       — called with each token string
 * @param {Function}      onDone        — called with final metadata object
 * @param {number}        topK
 * @param {string}        collection
 * @param {string|null}   conversationId
 * @returns {Function}    abort function to cancel the stream
 */
export function streamQuery(
  question,
  onToken,
  onDone,
  topK = 5,
  collection = 'default',
  conversationId = null,
) {
  const controller = new AbortController()

  ;(async () => {
    try {
      const res = await fetch(`${BASE_URL}/api/v1/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          top_k: topK,
          collection,
          conversation_id: conversationId,
        }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        onDone?.({ error: err.detail || 'Stream request failed' })
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const text = decoder.decode(value, { stream: true })
        const lines = text.split('\n')

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const payload = line.slice(6).trim()
          if (!payload) continue

          try {
            const data = JSON.parse(payload)
            if (data.token !== undefined) {
              onToken?.(data.token)
            } else if (data.done) {
              onDone?.({ chunks: data.retrieved_chunks })
            } else if (data.error) {
              onDone?.({ error: data.error })
            }
          } catch {
            // Ignore malformed SSE lines
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        onDone?.({ error: err.message })
      }
    }
  })()

  return () => controller.abort()
}

/**
 * Fetch the list of indexed documents plus vector store stats.
 * @param {string} collection
 */
export async function listDocuments(collection = 'default') {
  const response = await apiClient.get(
    `/api/v1/documents?collection=${encodeURIComponent(collection)}`,
  )
  return response.data
}

/**
 * List all collections with document and vector counts.
 */
export async function listCollections() {
  const response = await apiClient.get('/api/v1/collections')
  return response.data
}

/**
 * Start a new conversation session.
 * @returns {Promise<{ conversation_id: string }>}
 */
export async function startConversation() {
  const response = await apiClient.post('/api/v1/conversations/start')
  return response.data
}

/**
 * Fetch offline RAG evaluation metrics.
 */
export async function evaluateRag() {
  const response = await apiClient.get('/api/v1/evaluate')
  return response.data
}

export default apiClient
