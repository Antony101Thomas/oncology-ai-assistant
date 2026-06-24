"""
Agentic RAG orchestrator for Oncology AI.

execute_rag_query() is the single entry point that chains:
  1. Classify query  →  conceptual / precision / recency
  2. Hybrid retrieval from local vector store + BM25
  3. Conditionally call PubMed and ClinicalTrials.gov
  4. Build evidence context + citations
  5. Generate cited answer via Groq LLM
  6. Validate the answer (numeric guard + term overlap)
  7. Return structured result or a refusal
"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient

from bm25_search import keyword_search
from clinical_trials import search_clinical_trials
from embedder import embed_query
from pubmed_search import search_pubmed
from query_router import classify_query, should_call_live_apis
from validator import validate_answer

load_dotenv()

COLLECTION = "oncology_docs"

groq_client = Groq()


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    vector_results, bm25_results, k: int = 60
) -> list[dict[str, Any]]:
    scores: dict[int, dict[str, Any]] = {}

    for rank, result in enumerate(vector_results):
        chunk_id = result.id
        scores.setdefault(
            chunk_id,
            {"score": 0, "payload": result.payload, "id": chunk_id},
        )
        scores[chunk_id]["score"] += 1 / (k + rank + 1)

    for rank, result in enumerate(bm25_results):
        chunk_id = result["chunk_id"]
        scores.setdefault(
            chunk_id,
            {
                "score": 0,
                "payload": {
                    "text": result["text"],
                    "source": result["source"],
                    "page": result["page"],
                },
                "id": chunk_id,
            },
        )
        scores[chunk_id]["score"] += 1 / (k + rank + 1)

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:3]


# ── Confidence scoring ────────────────────────────────────────────────────────
#
# Scoring rubric:
#   Local PDF chunks found          → +2
#   3 or more PDF chunks found      → +1 bonus
#   2 PubMed results                → +2  |  1 result → +1
#   2 Trial results                 → +2  |  1 result → +1
#   Validator passed                → +1 bonus
#
# Thresholds:
#   high   → score >= 4  (PDFs + at least one live source)
#   medium → score >= 2  (PDFs only, or live sources only)
#   low    → score <  2  (very little evidence)

def _calculate_confidence(fused, pubmed, trials, validated: bool = False) -> str:
    score = 0

    if fused:
        score += 2
        if len(fused) >= 3:
            score += 1

    if len(pubmed) >= 2:
        score += 2
    elif len(pubmed) == 1:
        score += 1

    if len(trials) >= 2:
        score += 2
    elif len(trials) == 1:
        score += 1

    if validated:
        score += 1

    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


# ── Main orchestrator ─────────────────────────────────────────────────────────

def execute_rag_query(
    question: str,
    qdrant: QdrantClient,
    indexed_chunks: list[dict],
) -> dict[str, Any]:
    """
    Full agentic RAG pipeline for one oncology question.

    Args:
        question:       The user's natural-language question.
        qdrant:         Shared QdrantClient (must have COLLECTION already indexed).
        indexed_chunks: The list of chunks currently in the index.

    Returns a dict with keys:
        question, route, answer, confidence, citations,
        validated, validation_detail, refused
    """

    print(f"\n{'='*60}")
    print(f"[Agent] Question: {question}")

    # ── Stage 1: Classify ────────────────────────────────────────
    route = classify_query(question)
    print(f"[Agent] Route: {route}")

    # ── Stage 2: Local hybrid retrieval ──────────────────────────
    if indexed_chunks:
        query_vector = embed_query(question)
        vector_results = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=5,
        ).points
        bm25_results = keyword_search(question, top_k=5)
        fused = _reciprocal_rank_fusion(vector_results, bm25_results)
        print(f"[Agent] Local retrieval: {len(fused)} fused chunks")
    else:
        fused = []
        print("[Agent] No local index — skipping local retrieval")

    # ── Stage 3: Live API calls (routing-aware) ───────────────────
    call_live = should_call_live_apis(route, fused)
    print(f"[Agent] Call live APIs: {call_live}")

    if call_live:
        pubmed_results = search_pubmed(question, max_results=2)
        trial_results  = search_clinical_trials(question, max_results=2)
    else:
        pubmed_results = []
        trial_results  = []

    print(f"[Agent] PubMed: {len(pubmed_results)} | Trials: {len(trial_results)}")

    # ── Stage 4: Refusal check — no evidence at all ───────────────
    if not fused and not pubmed_results and not trial_results:
        print("[Agent] No evidence found — refusing")
        return {
            "question": question,
            "route": route,
            "answer": (
                "I don't have enough evidence to answer this question reliably. "
                "Please upload relevant oncology guidelines or try a different question."
            ),
            "confidence": "low",
            "citations": [],
            "validated": False,
            "validation_detail": {"valid": False, "status": "refused", "reason": "No evidence retrieved."},
            "refused": True,
        }

    # ── Stage 5: Build context + citations ───────────────────────
    pdf_context = "\n\n".join(r["payload"]["text"] for r in fused)
    pubmed_context = "\n\n".join(
        f"Title: {p['title']}\nAbstract: {p['abstract']}"
        for p in pubmed_results
    )
    trials_context = "\n\n".join(
        f"Trial: {t['title']}\nStatus: {t['status']}\nDescription: {t['description']}"
        for t in trial_results
    )

    combined_context = (
        f"From Medical Guidelines:\n{pdf_context}\n\n"
        f"From PubMed Research:\n{pubmed_context}\n\n"
        f"From Clinical Trials:\n{trials_context}"
    ).strip()

    citations = [
        {
            "title": f"{Path(r['payload']['source']).name} — Page {r['payload']['page']}",
            "source": "Uploaded oncology guideline PDF",
            "url": None,
            "type": "pdf",
        }
        for r in fused
    ]
    citations += [
        {"title": p["title"], "source": p["source"], "url": p["url"], "type": "pubmed"}
        for p in pubmed_results
    ]
    citations += [
        {"title": t["title"], "source": t["source"], "url": t["url"], "type": "trial"}
        for t in trial_results
    ]

    # ── Stage 6: Generate cited answer ───────────────────────────
    prompt = f"""You are an oncology medical assistant. You ONLY answer questions about cancer and oncology.

STRICT RULES:
- If the question is NOT about cancer, oncology, or cancer-related topics, you MUST respond with exactly: "This question is outside the scope of oncology. I can only answer cancer-related questions."
- Do NOT use cancer evidence to answer non-cancer questions (e.g. diabetes, heart disease, nutrition, general medicine).
- Only answer if the question is directly about cancer, tumors, oncology treatments, clinical trials, or cancer research.

If the question IS about oncology, format your answer clearly:
• Start with a one-sentence summary.
• Then use bullet points (•) for key facts, symptoms, treatments, or findings.
• Each bullet point should be on its own line.
• Keep each bullet concise and specific.
• Reference sources where relevant using [Source N].

Evidence:
{combined_context}

Question: {question}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    answer = response.choices[0].message.content
    print(f"[Agent] Answer generated ({len(answer.split())} words)")

    # ── Stage 7: Off-topic refusal check ─────────────────────────
    if "outside the scope of oncology" in answer.lower():
        return {
            "question": question,
            "route": route,
            "answer": "This question is outside the scope of oncology. I can only answer cancer-related questions.",
            "confidence": "low",
            "citations": [],
            "validated": False,
            "validation_detail": {"valid": False, "status": "refused", "reason": "Question is not oncology-related."},
            "refused": True,
        }

    # ── Stage 8: Validate ─────────────────────────────────────────
    all_sources = (
        [{"text": r["payload"]["text"]} for r in fused]
        + [{"text": p["abstract"]} for p in pubmed_results]
        + [{"text": t["description"]} for t in trial_results]
    )
    validation = validate_answer(answer, all_sources)
    print(f"[Agent] Validation: {validation['status']} | valid={validation['valid']}")

    # ── Stage 8: Final confidence (now includes validator result) ─
    confidence = _calculate_confidence(
        fused, pubmed_results, trial_results, validated=validation["valid"]
    )

    return {
        "question": question,
        "route": route,
        "answer": answer,
        "confidence": confidence,
        "citations": citations,
        "validated": validation["valid"],
        "validation_detail": validation,
        "refused": False,
    }
