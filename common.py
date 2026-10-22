"""Shared schema, taxonomy, validation and formatting for the fine-tuned request classifier."""
import re

REQUEST_TYPES = ["complaint", "general_enquiry", "service_request", "urgent_escalation"]
URGENCIES = ["low", "medium", "high", "critical"]
SENTIMENTS = ["positive", "neutral", "negative"]
LABEL_KEYS = ["request_type", "urgency", "sub_topic", "confidence", "summary",
              "customer_sentiment", "entities", "required_action"]

# Which urgencies we generate for each type. Rule used everywhere:
# urgency == "critical"  <=>  request_type == "urgent_escalation"
ALLOWED_URGENCY = {
    "complaint": ["medium", "high"],
    "general_enquiry": ["low", "medium"],
    "service_request": ["low", "medium", "high"],
    "urgent_escalation": ["critical"],
}
AMBIGUOUS_URGENCY = ["low", "medium"]

# Confidence target is assigned by clarity tier (NOT by the generating LLM).
CONFIDENCE_RANGE = {"clear": (0.85, 0.98), "ambiguous": (0.30, 0.60)}

SUB_TOPICS = {
    "complaint": ["wrong invoice amount", "double charge", "poor support experience",
                  "delayed service or delivery", "product quality problem",
                  "ticket unresolved for days", "rude agent", "unexpected price increase"],
    "general_enquiry": ["support hours", "pricing plans", "how to update billing info",
                        "refund policy", "feature availability", "integration availability",
                        "how to export data", "supported payment methods"],
    "service_request": ["change registered phone number", "change account email",
                        "update billing address", "add a user seat", "cancel subscription",
                        "request invoice copy", "upgrade plan", "update company name on account"],
    "urgent_escalation": ["account compromised", "unauthorized transactions",
                          "locked out of production account", "suspected data breach",
                          "outage affecting customers", "lost admin access",
                          "payment taken after cancellation with threat of legal action"],
}
TONES = ["polite", "neutral", "frustrated", "angry", "terse", "rambling", "formal", "anxious"]
LENGTHS = ["one short line", "two or three sentences", "one medium paragraph", "a long detailed paragraph"]
QUIRKS = ["none", "none", "none", "a few typos", "all lowercase, no punctuation",
          "includes an irrelevant side detail", "non-native English phrasing",
          "mentions a past interaction with support"]
HARD_NEG_ROUTINE = "uses urgent-sounding words like 'immediately' or 'ASAP' but the issue is routine"
HARD_NEG_URGENT = "calm and polite in tone despite a genuine emergency"

SYSTEM_PROMPT = (
    "You are a customer-request classifier. Read the request and reply with ONLY a JSON object "
    "with keys: request_type, urgency, sub_topic, confidence, summary, customer_sentiment, "
    "entities (invoice_number, amount, currency), required_action."
)


def build_user_message(subject: str, message: str) -> str:
    return f"Subject: {subject.strip()}\nMessage: {message.strip()}"


def to_chat_example(rec: dict) -> dict:
    import json
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(rec["subject"], rec["message"])},
        {"role": "assistant", "content": json.dumps(rec["label"], ensure_ascii=False)},
    ]}


NEG_WORDS = {"angry", "frustrated", "anxious", "worried", "upset", "annoyed", "concerned", "distressed", "urgent",
             "disappointed", "furious", "stressed", "panicked", "irritated", "dissatisfied", "displeased",
             "alarmed", "fearful", "scared", "desperate", "unhappy", "impatient", "fed up"}
POS_WORDS = {"happy", "pleased", "grateful", "satisfied", "appreciative", "thankful", "delighted", "excited",
             "optimistic", "hopeful", "cheerful", "positive"}
NEU_WORDS = {"calm", "formal", "terse", "curious", "confused", "inquisitive", "indifferent", "professional",
             "rambling", "polite", "uncertain", "neutral", "matter-of-fact", "courteous"}
CUR_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
CUR_CODES = ["USD", "EUR", "GBP", "INR", "AUD", "CAD"]
CUR_AMOUNT_RE = r"[$€£₹]\s?\d|\d\s?(?:USD|EUR|GBP|INR|AUD|CAD)\b|\b(?:USD|EUR|GBP|INR|AUD|CAD)\s?\d"


def sanitize_record(rec: dict):
    """Repair common small-model formatting slips deterministically. Returns (record, list_of_fixes).
    Entities are re-derived from the message where possible, so labels never contain invented values."""
    import copy
    rec = copy.deepcopy(rec)
    fixes, lab, msg = [], rec.get("label"), rec.get("message") or ""
    if not isinstance(lab, dict):
        return rec, fixes
    raw_s = lab.get("customer_sentiment", "")
    s = str(raw_s).strip().lower()
    new = s if s in SENTIMENTS else "negative" if s in NEG_WORDS else "positive" if s in POS_WORDS \
        else "neutral" if s in NEU_WORDS else None
    if new and new != raw_s:
        lab["customer_sentiment"] = new
        fixes.append("sentiment_lowercased" if s in SENTIMENTS else "sentiment_mapped")
    # vague messages have no reliable type: use one consistent best guess and a fixed sub_topic
    if rec.get("clarity") == "ambiguous":
        if lab.get("request_type") != "general_enquiry" or lab.get("sub_topic") != "unclear":
            fixes.append("ambiguous_normalised")
        lab["request_type"], lab["sub_topic"] = "general_enquiry", "unclear"
        if lab.get("urgency") not in AMBIGUOUS_URGENCY:
            lab["urgency"] = "low"
    ent = lab.get("entities")
    src = ent if isinstance(ent, dict) else {}
    if not isinstance(ent, dict) or set(ent) != {"invoice_number", "amount", "currency"}:
        fixes.append("entities_reshaped")
    inv = re.search(r"\bINV-\d+\b", msg)
    new_inv = inv.group(0) if inv else None
    amt = src.get("amount")
    new_amt = None
    if amt not in (None, ""):
        cleaned = re.sub(r"[^\d.]", "", str(amt).replace(",", ""))
        if re.fullmatch(r"\d+(\.\d+)?", cleaned) and cleaned in msg.replace(",", ""):
            new_amt = cleaned
    if new_amt is None:  # model missed it: recover only when the message has exactly ONE currency-marked amount
        found = {re.sub(r"[^\d.]", "", m.replace(",", "")) for m in
                 re.findall(r"(?:[$€£₹]\s?|\b(?:USD|EUR|GBP|INR|AUD|CAD)\s?)(\d[\d,]*(?:\.\d+)?)", msg)}
        found |= {re.sub(r"[^\d.]", "", m.replace(",", "")) for m in
                  re.findall(r"(\d[\d,]*(?:\.\d+)?)\s?(?:USD|EUR|GBP|INR|AUD|CAD)\b", msg)}
        found = {f.rstrip(".") for f in found if f}
        if len(found) == 1:
            new_amt = found.pop(); fixes.append("amount_recovered")
    new_cur = None
    if new_amt is not None:
        for sym, iso in CUR_SYMBOLS.items():
            if sym in msg:
                new_cur = iso; break
        if new_cur is None:
            for code in CUR_CODES:
                if re.search(rf"\b{code}\b", msg, re.I):
                    new_cur = code; break
    new_ent = {"invoice_number": new_inv, "amount": new_amt, "currency": new_cur}
    if new_ent != ent:
        if isinstance(ent, dict) and set(ent) == set(new_ent):
            fixes.append("entities_corrected")
        lab["entities"] = new_ent
    return rec, fixes


def validate_record(rec: dict) -> list:
    """Return a list of problems (empty list = valid)."""
    errs = []
    subj, msg, lab = rec.get("subject"), rec.get("message"), rec.get("label")
    if not isinstance(subj, str) or not subj.strip():
        errs.append("empty subject")
    if not isinstance(msg, str) or not (5 <= len(msg) <= 2500):
        errs.append("bad message length")
    elif rec.get("clarity") == "ambiguous" and len(msg.split()) > 16:
        errs.append("ambiguous but message too long/specific")
    if not isinstance(lab, dict) or set(lab) != set(LABEL_KEYS):
        return errs + ["label keys mismatch"]
    if lab["request_type"] not in REQUEST_TYPES:
        errs.append("bad request_type")
    if lab["urgency"] not in URGENCIES:
        errs.append("bad urgency")
    if (lab["urgency"] == "critical") != (lab["request_type"] == "urgent_escalation"):
        errs.append("critical/urgent_escalation mismatch")
    if lab["customer_sentiment"] not in SENTIMENTS:
        errs.append("bad sentiment")
    if not isinstance(lab["confidence"], (int, float)) or not 0 <= lab["confidence"] <= 1:
        errs.append("bad confidence")
    for k in ("sub_topic", "summary", "required_action"):
        if not isinstance(lab[k], str) or not lab[k].strip():
            errs.append(f"empty {k}")
    ent = lab["entities"]
    if not isinstance(ent, dict) or set(ent) != {"invoice_number", "amount", "currency"}:
        return errs + ["entities keys mismatch"]
    text = (msg or "")
    if ent["amount"] is not None:
        if not isinstance(ent["amount"], str) or not re.fullmatch(r"\d+(\.\d+)?", ent["amount"]):
            errs.append("amount not plain number string")
        elif ent["amount"] not in text.replace(",", ""):
            errs.append("amount not found in message (invented?)")
    if ent["invoice_number"] is not None and str(ent["invoice_number"]) not in text:
        errs.append("invoice_number not found in message (invented?)")
    if ent["currency"] is not None and not re.fullmatch(r"[A-Z]{3}", str(ent["currency"])):
        errs.append("currency not ISO code")
    if re.search(r"\]\((?:mailto:|https?:)", text):
        errs.append("markdown link artifact in message")
    if ent["invoice_number"] is None and re.search(r"invoice\s*(?:number|no\.?|#)?\s*[:#]?\s*\d{4,}", text, re.I):
        errs.append("invoice number without INV- prefix (entity would be wrong)")
    if re.search(r"\bINV-\d+\b", text) and ent["invoice_number"] is None:
        errs.append("message has an invoice number but entity is null")
    if re.search(CUR_AMOUNT_RE, text) and ent["amount"] is None:
        errs.append("message has a currency amount but entity is null")
    return errs