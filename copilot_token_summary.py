"""
Parses a GitHub Copilot Chat OTel export (copilot-otel.jsonl) and summarizes
token usage per session and per model.

Usage:
    python copilot_token_summary.py "C:\\Users\\<you>\\copilot-otel.jsonl"
"""

import json
import sys
from collections import defaultdict


def load_events(path):
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [skip] Could not parse line {line_num}")
    return events


def summarize(events):
    # per_session[session_id][model] = {"input": x, "output": y, "calls": n}
    per_session = defaultdict(lambda: defaultdict(lambda: {"input": 0, "output": 0, "calls": 0}))
    overall = defaultdict(lambda: {"input": 0, "output": 0, "calls": 0})

    for ev in events:
        attrs = ev.get("attributes", {})
        event_name = attrs.get("event.name", "")

        # Only inference events carry token usage
        if event_name != "gen_ai.client.inference.operation.details":
            continue

        model = attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model") or "unknown"
        input_tokens = attrs.get("gen_ai.usage.input_tokens", 0)
        output_tokens = attrs.get("gen_ai.usage.output_tokens", 0)

        # session.id lives in resource._rawAttributes as a ["session.id", "<value>"] pair
        session_id = "unknown"
        raw_attrs = ev.get("resource", {}).get("_rawAttributes", [])
        for pair in raw_attrs:
            if isinstance(pair, list) and len(pair) == 2 and pair[0] == "session.id":
                session_id = pair[1]
                break

        bucket = per_session[session_id][model]
        bucket["input"] += input_tokens
        bucket["output"] += output_tokens
        bucket["calls"] += 1

        ov = overall[model]
        ov["input"] += input_tokens
        ov["output"] += output_tokens
        ov["calls"] += 1

    return per_session, overall


def print_report(per_session, overall):
    print("\n=== Per Session / Per Model ===")
    for session_id, models in per_session.items():
        print(f"\nSession: {session_id}")
        session_total_in = session_total_out = session_calls = 0
        for model, usage in models.items():
            total = usage["input"] + usage["output"]
            print(f"  {model:35s} calls={usage['calls']:3d}  in={usage['input']:7,d}  out={usage['output']:6,d}  total={total:7,d}")
            session_total_in += usage["input"]
            session_total_out += usage["output"]
            session_calls += usage["calls"]
        print(f"  {'-> session total':35s} calls={session_calls:3d}  in={session_total_in:7,d}  out={session_total_out:6,d}  total={session_total_in+session_total_out:7,d}")

    print("\n=== Overall (All Sessions) By Model ===")
    grand_in = grand_out = grand_calls = 0
    for model, usage in overall.items():
        total = usage["input"] + usage["output"]
        print(f"  {model:35s} calls={usage['calls']:3d}  in={usage['input']:7,d}  out={usage['output']:6,d}  total={total:7,d}")
        grand_in += usage["input"]
        grand_out += usage["output"]
        grand_calls += usage["calls"]

    print(f"\n=== Grand Total ===")
    print(f"  Calls: {grand_calls}")
    print(f"  Input tokens:  {grand_in:,}")
    print(f"  Output tokens: {grand_out:,}")
    print(f"  Total tokens:  {grand_in + grand_out:,}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python copilot_token_summary.py <path-to-copilot-otel.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    events = load_events(path)
    print(f"Loaded {len(events)} events from {path}")

    per_session, overall = summarize(events)
    print_report(per_session, overall)
