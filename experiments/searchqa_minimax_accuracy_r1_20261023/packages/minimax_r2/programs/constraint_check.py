"""Check exact source quotes and visible textual constraints without semantic grading."""
import re

_NUMBERS = {name: value for value, name in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_NUMBER = r"(?:[1-9][0-9]?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"


def _documents(text):
    starts = [match.start() for match in re.finditer(r"\[DOC\]", text)]
    if not starts:
        starts = [0]
    elif text[:starts[0]].strip():
        starts.insert(0, 0)
    return [(start, starts[i + 1] if i + 1 < len(starts) else len(text)) for i, start in enumerate(starts)]


def _occurrences(text, value, documents):
    if not value:
        return []
    result = []
    for match in re.finditer(re.escape(value), text):
        number = next((i for i, (start, end) in enumerate(documents) if start <= match.start() and match.end() <= end), None)
        result.append({"span": [match.start(), match.end()], "doc_id": number, "within_single_doc": number is not None})
        if len(result) >= 12:
            break
    return result


def run(ctx, inputs):
    question, context, answer = inputs.get("question"), inputs.get("context"), inputs.get("answer")
    quotes = inputs.get("quotes", [])
    if not all(isinstance(value, str) for value in (question, context, answer)) or not answer.strip():
        return {"status": "needs_input", "outputs": {}, "error": "question/context strings and a nonempty answer are required"}
    if not isinstance(quotes, list) or len(quotes) > 8 or any(not isinstance(quote, str) or not quote or len(quote) > 1200 for quote in quotes):
        return {"status": "needs_input", "outputs": {}, "error": "quotes must contain at most 8 nonempty strings of at most 1200 characters"}
    documents, quote_checks, flags = _documents(context), [], []
    for quote in quotes:
        matches = _occurrences(context, quote, documents)
        status = "exact_single_doc" if any(item["within_single_doc"] for item in matches) else "cross_document" if matches else "not_found"
        quote_checks.append({"quote": quote, "status": status, "occurrences": matches})
        if status != "exact_single_doc":
            flags.append({"kind": "quote_" + status, "quote_index": len(quote_checks) - 1})
    constraints = []
    for unit in ("letter", "word"):
        pattern = re.compile(r"\b(?P<n>" + _NUMBER + r")[ -]+" + unit + r"(?:ed)?\s+(?:word|name|term|answer|phrase)\b", re.I)
        for match in pattern.finditer(question):
            raw = match.group("n").lower()
            expected = int(raw) if raw.isdigit() else _NUMBERS[raw]
            observed = sum(char.isalpha() for char in answer) if unit == "letter" else len(re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", answer))
            constraints.append({"clue_span": [match.start(), match.end()], "clue_text": match.group(),
                                "kind": unit + "_count", "expected": expected, "observed": observed,
                                "literal_match": expected == observed, "scope": "unverified_may_describe_an_anchor"})
            if expected != observed:
                flags.append({"kind": "possible_" + unit + "_count_mismatch", "constraint_index": len(constraints) - 1})
    occurrences = _occurrences(context, answer, documents)
    report = {"answer": answer, "answer_exact_occurrences": occurrences, "quote_checks": quote_checks,
              "visible_length_constraints": constraints, "flags": flags,
              "semantic_status": "not_checked", "answer_presence_is_required": False,
              "offset_unit": "unicode_character_end_exclusive",
              "meaning": "Reconsider a specific flagged claim only. Literal presence does not prove the answer; absence can reflect a valid alias. Constraint scope still needs semantic judgment."}
    return {"status": "ok", "outputs": {"constraint_check": report}}
