"""
End-to-end RAG pipeline for a support ticket:
1. Retrieve relevant KB chunks
2. Retrieve similar past resolved tickets
3. Pull account context if available
4. Build a grounded suggested-response prompt (LLM call is a stub —
   plug in Anthropic API / OpenAI / whatever you're allowed to use)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_tickets, load_accounts, build_account_map, ticket_text_for_embedding
from kb_index_tfidf import load_tfidf_index, search_tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_ticket_similarity_index(tickets: list[dict]):
    """TF-IDF over resolved/closed tickets, for 'similar past ticket' retrieval."""
    resolved = [t for t in tickets if t["status"] in ("Resolved", "Closed")]
    texts = [ticket_text_for_embedding(t) for t in resolved]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, resolved


def find_similar_tickets(query_text, vectorizer, matrix, resolved_tickets, top_k=3):
    q_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return [(resolved_tickets[i], float(sims[i])) for i in top_idx]


def suggest_response(ticket: dict, kb_vectorizer, kb_matrix, kb_chunks,
                      ticket_vectorizer, ticket_matrix, resolved_tickets,
                      account_map: dict) -> dict:
    query_text = ticket_text_for_embedding(ticket)

    kb_hits = search_tfidf(query_text, kb_vectorizer, kb_matrix, kb_chunks, top_k=3)
    similar_tickets = find_similar_tickets(
        query_text, ticket_vectorizer, ticket_matrix, resolved_tickets, top_k=3
    )
    account = account_map.get(ticket["account_id"])

    # ---- Build the grounded context that would go to an LLM ----
    context_parts = ["## Relevant KB articles"]
    for chunk, score in kb_hits:
        context_parts.append(f"- [{chunk.heading}] (score={score:.2f})\n  {chunk.text[:300]}")

    context_parts.append("\n## Similar past resolved tickets")
    for t, score in similar_tickets:
        context_parts.append(
            f"- {t['ticket_id']} (score={score:.2f}): {t['subject']}"
        )

    if account:
        context_parts.append("\n## Account context")
        context_parts.append(
            f"- {account['company']}, {account['plan_tier']} plan, "
            f"health: {account['health_status']}, trend: {account['usage_trend']}"
        )
        if account.get("escalation_notes"):
            context_parts.append(f"- Escalation notes: {'; '.join(account['escalation_notes'])}")
    else:
        context_parts.append("\n## Account context")
        context_parts.append("- No matching account record found for this ticket's account_id.")

    grounded_context = "\n".join(context_parts)

    # ---- LLM call stub ----
    # In your real submission, send `grounded_context` + the ticket text to
    # an LLM (Claude/GPT) with a prompt like:
    #   "You are a support agent. Using ONLY the context below, draft a
    #    reply to the customer's ticket. Cite which KB article you used."
    # For this pipeline demo we just return the assembled context.

    return {
        "ticket_id": ticket["ticket_id"],
        "grounded_context": grounded_context,
        "kb_hits": [(c.heading, s) for c, s in kb_hits],
        "similar_tickets": [(t["ticket_id"], s) for t, s in similar_tickets],
        "account_matched": account is not None,
    }


if __name__ == "__main__":
    tickets = load_tickets()
    accounts = load_accounts()
    account_map = build_account_map(accounts)

    kb_vectorizer, kb_matrix, kb_chunks = load_tfidf_index()
    ticket_vectorizer, ticket_matrix, resolved_tickets = build_ticket_similarity_index(tickets)

    # demo on the first open/in-progress ticket
    demo_ticket = next(t for t in tickets if t["status"] in ("Open", "In Progress"))
    result = suggest_response(
        demo_ticket, kb_vectorizer, kb_matrix, kb_chunks,
        ticket_vectorizer, ticket_matrix, resolved_tickets, account_map
    )

    print(f"Ticket: {demo_ticket['ticket_id']} — {demo_ticket['subject']}\n")
    print(result["grounded_context"])
