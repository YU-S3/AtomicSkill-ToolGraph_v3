"""Extract literal matching fields from current public input; no answer memory."""
import re

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
    if not isinstance(question, str) or not isinstance(context, str):
        return {"status": "needs_input", "outputs": {}, "error": "question/context strings required"}
    fields = _explicit_fields(context, question, _documents(context))
    for field in fields:
        field["clue_text"] = context[slice(*field["clue_span"])]
    return {"status": "ok", "outputs": {"source_fields": {
        "fields": fields[:4], "omitted": max(0,len(fields)-4),
        "status": "matched_literal_fields" if fields else "no_matching_field",
        "offset_unit": "unicode_character_end_exclusive",
        "semantic_status": "unverified; check relation, qualifiers and requested scope"}}}
