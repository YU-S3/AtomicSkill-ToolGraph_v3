"""Index supplied SearchQA document boundaries; offsets are Unicode character offsets."""
import re

_STOP = set("a an the of in on at to for from and or by with is was are were be been this that these those it its as who what which when where how his her their has have had not one named called".split())


def _tokens(value):
    return {word.casefold() for word in re.findall(r"[^\W_]+", value) if len(word) > 1 and word.casefold() not in _STOP}


def _documents(text):
    starts = [match.start() for match in re.finditer(r"\[DOC\]", text)]
    if not starts:
        starts = [0]
    elif text[:starts[0]].strip():
        starts.insert(0, 0)
    records = []
    for number, start in enumerate(starts):
        end = starts[number + 1] if number + 1 < len(starts) else len(text)
        raw = text[start:end]
        title_marker, paragraph_marker = raw.find("[TLE]"), raw.find("[PAR]")
        title_start = start + title_marker + 5 if title_marker >= 0 else start
        title_end = start + paragraph_marker if paragraph_marker >= 0 else end
        if title_marker < 0 or title_end < title_start:
            title_start = title_end = start
        while title_start < title_end and text[title_start].isspace():
            title_start += 1
        while title_end > title_start and text[title_end - 1].isspace():
            title_end -= 1
        body_start = start + paragraph_marker + 5 if paragraph_marker >= 0 else start
        while body_start < end and text[body_start].isspace():
            body_start += 1
        records.append({"doc_id": number, "start": start, "end": end,
                        "title_start": title_start, "title_end": title_end, "body_start": body_start})
    return records


def run(ctx, inputs):
    question, context = inputs.get("question"), inputs.get("context")
    if not isinstance(question, str) or not isinstance(context, str):
        return {"status": "needs_input", "outputs": {}, "error": "question and context must be strings"}
    documents, clue = _documents(context), _tokens(question)
    seen, result = {}, []
    for doc in documents:
        raw = context[doc["start"]:doc["end"]]
        key = " ".join(raw.casefold().split())
        duplicate = seen.get(key)
        seen.setdefault(key, doc["doc_id"])
        preview_end = min(doc["end"], doc["body_start"] + 160)
        result.append({"doc_id": doc["doc_id"], "span": [doc["start"], doc["end"]],
                       "title": context[doc["title_start"]:doc["title_end"]],
                       "title_span": [doc["title_start"], doc["title_end"]],
                       "preview": context[doc["body_start"]:preview_end],
                       "preview_span": [doc["body_start"], preview_end],
                       "clue_terms_present": sorted(clue & _tokens(raw))[:32],
                       "duplicate_of": duplicate})
    report = {"context_chars": len(context), "document_count": len(documents), "documents": result[:64],
              "omitted_documents": max(0, len(result) - 64), "offset_unit": "unicode_character_end_exclusive",
              "meaning": "Source locations and literal clue overlap only; not semantic relevance or independent corroboration."}
    return {"status": "ok", "outputs": {"context_index": report}}
