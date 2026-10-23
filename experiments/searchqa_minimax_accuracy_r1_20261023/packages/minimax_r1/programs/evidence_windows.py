"""Return exact bounded source windows and conservatively detected explicit QA fields.

Lexical ranking is navigation, not an answer selector. No network, labels or lookup table.
"""
import math
import re

_STOP = set("a an the of in on at to for from and or by with is was are were be been this that these those it its as who what which when where how his her their has have had not one named called".split())


def _tokens(value):
    return {word.casefold() for word in re.findall(r"[^\W_]+", value) if len(word) > 1 and word.casefold() not in _STOP}


def _norm(value):
    return " ".join(re.findall(r"[^\W_]+", value.casefold()))


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


def _explicit_fields(text, question, documents):
    fields, target = [], _norm(question)
    if not target:
        return fields
    labelled = re.compile(r"(?:^|\n)[ \t]*(?:Q(?:uestion)?|Clue)[ \t]*:[ \t]*(?P<q>[^\n]+?)"
                          r"(?:[ \t]+|\n[ \t]*)(?:A(?:nswer)?)[ \t]*:[ \t]*(?P<a>[^\n]+)", re.I)
    for doc in documents:
        body = text[doc["body_start"]:doc["end"]]
        matches = []
        for match in labelled.finditer(body):
            matches.append((match.group("q"), match.span("q"), match.group("a"), match.span("a"), "explicit_Q_A_labels"))
        cursor = 0
        for line in body.splitlines(keepends=True):
            cells = list(re.finditer(r"[^|\t\r\n]+", line))
            if "|" in line or "\t" in line:
                for left, right in zip(cells, cells[1:]):
                    matches.append((left.group(), (cursor + left.start(), cursor + left.end()),
                                    right.group(), (cursor + right.start(), cursor + right.end()), "explicit_pipe_or_tab_fields"))
            cursor += len(line)
        for clue, clue_span, answer, answer_span, form in matches:
            if _norm(clue) != target or not answer.strip() or len(answer.strip()) > 256:
                continue
            left = len(answer) - len(answer.lstrip())
            right = len(answer.rstrip())
            start, end = doc["body_start"] + answer_span[0] + left, doc["body_start"] + answer_span[0] + right
            if any(field["answer_span"] == [start, end] for field in fields):
                continue
            fields.append({"doc_id": doc["doc_id"], "candidate": text[start:end], "answer_span": [start, end],
                           "clue_span": [doc["body_start"] + clue_span[0], doc["body_start"] + clue_span[1]],
                           "boundary_evidence": form, "clue_match": "all_normalized_words_in_order",
                           "semantic_status": "unverified; check requested type, relation, and answer form"})
    return fields


def run(ctx, inputs):
    question, context = inputs.get("question"), inputs.get("context")
    query, top_k = inputs.get("query", question), inputs.get("top_k", 4)
    if not isinstance(question, str) or not isinstance(context, str) or not isinstance(query, str):
        return {"status": "needs_input", "outputs": {}, "error": "question, context and query must be strings"}
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 6:
        return {"status": "needs_input", "outputs": {}, "error": "top_k must be an integer from 1 to 6"}
    documents, terms = _documents(context), _tokens(query)
    doc_terms = [_tokens(context[doc["start"]:doc["end"]]) for doc in documents]
    weights = {term: math.log(1 + (len(documents) + 1) / (1 + sum(term in group for group in doc_terms))) for term in terms}
    candidates = []
    for doc in documents:
        title = context[doc["title_start"]:doc["title_end"]]
        title_hits = terms & _tokens(title)
        body_start, end = doc["body_start"], doc["end"]
        starts = list(range(body_start, max(body_start + 1, end), 320))
        for start in starts:
            stop = min(end, start + 560)
            quote = context[start:stop]
            hits = terms & _tokens(quote)
            score = sum(weights[term] for term in hits) + 0.35 * sum(weights[term] for term in title_hits)
            candidates.append({"doc_id": doc["doc_id"], "span": [start, stop], "quote": quote,
                               "title": title, "title_span": [doc["title_start"], doc["title_end"]],
                               "matched_terms": sorted(hits | title_hits), "lexical_score": round(score, 6)})
    candidates.sort(key=lambda item: (-item["lexical_score"], item["doc_id"], item["span"][0]))
    windows, seen, per_doc = [], set(), {}
    for candidate in candidates:
        key = " ".join(candidate["quote"].casefold().split())
        if not key or key in seen or per_doc.get(candidate["doc_id"], 0) >= 2:
            continue
        seen.add(key)
        per_doc[candidate["doc_id"]] = per_doc.get(candidate["doc_id"], 0) + 1
        windows.append(candidate)
        if len(windows) >= top_k:
            break
    fields = _explicit_fields(context, question, documents)
    report = {"query": query, "windows": windows, "explicit_qa_fields": fields[:6],
              "omitted_qa_fields": max(0, len(fields) - 6), "source_document_count": len(documents),
              "offset_unit": "unicode_character_end_exclusive", "searched_source": "only_supplied_context",
              "meaning": "Lexical navigation and explicit field boundaries, not semantic proof, gold prediction, or a calibrated confidence score."}
    return {"status": "ok", "outputs": {"evidence_windows": report}}
