/**
 * apiClient.js
 * Centralised Axios instance for all backend communication.
 *
 * Exports two named helpers:
 *   uploadDocument(file, onUploadProgress)  → POST /api/v1/upload
 *   queryDocuments(question, topK)          → POST /api/v1/query
 *
 * The base URL is read from the VITE_API_BASE_URL env variable so the
 * same build can target different environments.  Falls back to the local
 * FastAPI server (http://localhost:8000) when the variable is unset.
 */

import axios from 'axios'

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 120_000, // 2-minute timeout — LLM generation can be slow
  headers: {
    Accept: 'application/json',
  },
})

// ---------------------------------------------------------------------------
// Response interceptor — uniform error shaping
// ---------------------------------------------------------------------------

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Prefer the backend's detail message; fall back to the HTTP status text
    const detail =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred.'

    // Attach a clean message so callers can use error.userMessage
    error.userMessage = detail
    return Promise.reject(error)
  },
)

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

/**
 * Upload a single document to the backend for indexing.
 *
 * @param {File}     file               - The File object to upload.
 * @param {Function} onUploadProgress   - Axios progress callback
 *                                        ({ loaded, total, progress }).
 * @returns {Promise<{ filename, chunks_indexed, message }>}
 */
export async function uploadDocument(file, onUploadProgress) {
  const form = new FormData()
  form.append('file', file)

  const response = await apiClient.post('/api/v1/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
  })

  return response.data
}

/**
 * Send a question to the RAG pipeline and receive a generated answer.
 *
 * @param {string} question - Natural-language question.
 * @param {number} topK     - Number of context chunks to retrieve (default 5).
 * @returns {Promise<{ question, answer, retrieved_chunks, processing_time_seconds }>}
 */
export async function queryDocuments(question, topK = 5) {
  const response = await apiClient.post('/api/v1/query', {
    question,
    top_k: topK,
  })
  return response.data
}

/**
 * Fetch the list of indexed documents plus the vector store stats.
 *
 * @returns {Promise<{ documents, total_documents, total_vectors_indexed }>}
 */
export async function listDocuments() {
  const response = await apiClient.get('/api/v1/documents')
  return response.data
}

export default apiClient
