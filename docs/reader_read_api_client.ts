/** Minimal read-only browser adapter. No agent submission or inference occurs here. */
export type Relation = "supports" | "contradicts" | "qualifies" | "context" | "insufficient_support" | null;
export interface Page { offset: number; limit: number; total: number; has_more: boolean; next_offset: number | null; }
export interface Identity { api_version: string; paper_id: string; paper_revision: string; snapshot_id: string; }
export interface GraphEdge {
  from: string; to: string; mapping_id: string; relation: Relation;
  relation_status: string; explanation: string; source_parts_used: string[];
  source_text_role_for_target?: string; origin_note?: string;
}
export interface ClaimView extends Identity {
  claim: { claim_id: string; text: string; source_anchor: unknown; current_mapping_id: string | null; };
  mapping_status: "not_analyzed" | "unavailable" | "available";
  mapping: null | {
    mapping_id: string; claim_id: string; explanation: string;
    material_status: string; qualifications: string[]; evidence_ids_used: string[];
    historical_explanation_requires_review: boolean; presentation_review_notes: string[];
  };
  evidence: Record<string, { evidence_id: string; preview: { text: string; is_complete: boolean }; detail_url: string; }>;
  display: null | {
    claim_id: string; mapping_id: string; evidence_ids: string[];
    evidence_highlights: Array<{ evidence_id: string; source_anchor: unknown; source_parts_used: string[] }>;
    nodes: Array<{ id: string; kind: string; source_element_id?: string }>;
    edges: GraphEdge[]; graph_complete: boolean;
  };
  evidence_page?: Page; next_evidence_url?: string | null; issues: string[]; warnings: string[];
}

export class ReaderReadClient {
  private readonly base: string;
  constructor(apiBase = "http://127.0.0.1:8010/api") { this.base = apiBase.replace(/\/$/, ""); }

  private async fetchJSON<T>(url: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(url, { credentials: "omit", ...init });
    const payload = await response.json();
    if (!response.ok || payload.status !== "success") {
      const error = new Error(payload.error?.message ?? `Read failed (${response.status})`);
      Object.assign(error, { code: payload.error?.code, status: response.status, details: payload.error?.details });
      throw error;
    }
    return payload.data as T;
  }

  overview(paperId: string) {
    return this.fetchJSON<Identity & { routes: Record<string, string> } & Record<string, unknown>>(`${this.base}/papers/${encodeURIComponent(paperId)}`);
  }

  claims(paperId: string, offset = 0, limit = 50, snapshotId?: string) {
    const query = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (snapshotId) query.set("snapshot_id", snapshotId);
    return this.fetchJSON<Identity & { items: unknown[]; page: Page }>(`${this.base}/papers/${encodeURIComponent(paperId)}/claims?${query}`);
  }

  claim(paperId: string, claimId: string, mappingId?: string) {
    const query = mappingId ? `?mapping_id=${encodeURIComponent(mappingId)}` : "";
    return this.fetchJSON<ClaimView>(`${this.base}/papers/${encodeURIComponent(paperId)}/claims/${encodeURIComponent(claimId)}${query}`);
  }

  /** Follow server-bound detail/page URLs; never concatenate internal disk paths. */
  follow<T>(resourceUrl: string) {
    const base = new URL(this.base, window.location.href);
    const target = new URL(resourceUrl, base);
    if (target.origin !== base.origin || !target.pathname.startsWith(`${base.pathname}/`)) {
      throw new Error("Refusing a URL outside the configured reader API.");
    }
    return this.fetchJSON<T>(target.href);
  }

  response(paperId: string, workflowId?: string) {
    const query = workflowId ? `?workflow_id=${encodeURIComponent(workflowId)}` : "";
    return this.fetchJSON<Identity & Record<string, unknown>>(`${this.base}/papers/${encodeURIComponent(paperId)}/response${query}`);
  }

  lookupSelection(paperId: string, body: {
    paper_revision: string; segments: Array<{element_id: string; start: number; end: number; exact: string}>;
    offset?: number; limit?: number; snapshot_id?: string;
  }) {
    return this.fetchJSON<Identity & { items: unknown[]; page: Page; new_extraction: false }>(
      `${this.base}/papers/${encodeURIComponent(paperId)}/claims/selection`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  }
}

/** Null labels must remain neutral; no `relation || "supports"` fallback. */
export function relationLabel(edge: GraphEdge): string {
  if (edge.relation_status === "needs_origin_review") return "Relation needs review";
  if (edge.relation === null) return "Relation not recorded";
  return edge.relation;
}

/** Convert a DOM UTF-16 offset in a canonical text node to a code-point index. */
export function codePointOffset(text: string, utf16Offset: number): number {
  if (!Number.isInteger(utf16Offset) || utf16Offset < 0 || utf16Offset > text.length) throw new Error("Invalid text offset");
  if (utf16Offset > 0 && utf16Offset < text.length &&
      /[\uD800-\uDBFF]/.test(text[utf16Offset - 1]) && /[\uDC00-\uDFFF]/.test(text[utf16Offset])) {
    throw new Error("A selection boundary splits a Unicode surrogate pair");
  }
  return Array.from(text.slice(0, utf16Offset)).length;
}
