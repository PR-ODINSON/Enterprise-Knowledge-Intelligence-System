/**
 * CollectionSelector
 * ─────────────────────────────────────────────────────────
 * Dropdown that lets the user choose an existing collection
 * or type a new one to create on first document upload.
 *
 * Props:
 *   collections  (Array)  — list of { collection_id, total_documents, total_vectors } objects
 *   selected     (string) — currently active collection_id
 *   onChange     (fn)     — called with new collection_id string
 *   loading      (bool)   — show skeleton while collections load
 */
export default function CollectionSelector({ collections = [], selected, onChange, loading }) {
  const handleCreate = () => {
    const name = prompt('New collection name (lowercase, no spaces):')
    if (!name) return
    const slug = name.trim().toLowerCase().replace(/\s+/g, '-')
    if (slug) onChange?.(slug)
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2">
        <div className="h-8 w-36 animate-pulse rounded-lg bg-slate-200" />
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2">
      <FolderIcon className="h-4 w-4 shrink-0 text-slate-400" />
      <select
        id="collection-selector"
        value={selected}
        onChange={(e) => onChange?.(e.target.value)}
        className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-100"
      >
        {collections.map((c) => (
          <option key={c.collection_id} value={c.collection_id}>
            {c.collection_id} ({c.total_documents} doc{c.total_documents !== 1 ? 's' : ''})
          </option>
        ))}
      </select>
      <button
        type="button"
        onClick={handleCreate}
        title="Create new collection"
        className="flex h-7 w-7 items-center justify-center rounded-lg border border-dashed border-slate-300 text-slate-400 hover:border-brand-400 hover:text-brand-500 transition-colors"
      >
        <PlusIcon className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

function FolderIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M2.25 12.75V12A2.25 2.25 0 014.5 9.75h15A2.25 2.25 0 0121.75 12v.75m-8.69-6.44l-2.12-2.12a1.5 1.5 0 00-1.061-.44H4.5A2.25 2.25 0 002.25 6v12a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9a2.25 2.25 0 00-2.25-2.25h-5.379a1.5 1.5 0 01-1.06-.44z" />
    </svg>
  )
}

function PlusIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
    </svg>
  )
}
