"""
Data loading utilities for the support-assistant project.
Handles the tickets <-> accounts join, including the many tickets
whose account_id has no matching record in accounts.json.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_tickets(path: Path = DATA_DIR / "tickets.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_accounts(path: Path = DATA_DIR / "accounts.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_account_map(accounts: list[dict]) -> dict[str, dict]:
    """account_id -> account record, for O(1) lookup."""
    return {a["account_id"]: a for a in accounts}


def enrich_ticket(ticket: dict, account_map: dict[str, dict]) -> dict:
    """
    Attach account context to a ticket where available.
    Sets `account` to None (not an exception) when there's no match —
    this is the expected case for ~99% of tickets in this dataset.
    """
    account = account_map.get(ticket["account_id"])
    return {**ticket, "account": account}


def enrich_all_tickets(tickets: list[dict], accounts: list[dict]) -> list[dict]:
    account_map = build_account_map(accounts)
    return [enrich_ticket(t, account_map) for t in tickets]


def ticket_text_for_embedding(ticket: dict) -> str:
    """Concatenate the fields that matter for semantic search."""
    parts = [
        ticket.get("subject", ""),
        ticket.get("body", ""),
        f"Product: {ticket.get('product', '')}",
        f"Area: {ticket.get('product_area', '')}",
        f"Category: {ticket.get('category', '')}",
    ]
    return "\n".join(p for p in parts if p)


if __name__ == "__main__":
    tickets = load_tickets()
    accounts = load_accounts()
    enriched = enrich_all_tickets(tickets, accounts)

    matched = sum(1 for t in enriched if t["account"] is not None)
    print(f"Loaded {len(tickets)} tickets, {len(accounts)} accounts")
    print(f"Tickets with matched account: {matched}/{len(tickets)} "
          f"({matched/len(tickets):.1%})")
    print("\nSample enriched ticket (matched, if any found):")
    for t in enriched:
        if t["account"] is not None:
            print(json.dumps(t, indent=2)[:800])
            break
