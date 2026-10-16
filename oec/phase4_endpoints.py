"""Endpoint contract: bounded FINAL reader failure is zero; structural failures block."""


def index_rows(rows, key):
    output = {row[key]: row for row in rows}
    if not rows or len(output) != len(rows):
        raise ValueError("Empty or duplicate ids")
    return output


def paired_points(before, after, gold, exact_match):
    before, after, gold = index_rows(before, "task_id"), index_rows(after, "task_id"), index_rows(gold, "_id")
    if set(before) != set(after) or set(before) != set(gold):
        raise ValueError("All assigned task ids must match exactly")
    def score(row, expected):
        if type(row.get("runtime_valid")) is not bool or not isinstance(row.get("answer"), str):
            raise ValueError("Invalid reader result contract")
        if row.get("failure_stage") == "upstream_selection":
            raise ValueError("Upstream selection failure blocks inference")
        if not row["runtime_valid"]:
            if row.get("failure_stage") != "bounded_reader_output":
                raise ValueError("Structural failures cannot be scored as zero")
            return 0
        if row.get("status") == "INVALID_RUNTIME_EVALUATION":
            raise ValueError("Contradictory valid/invalid flags")
        return int(exact_match(row["answer"], expected["answer"]))
    return [(score(before[key], gold[key]), score(after[key], gold[key])) for key in sorted(gold)]
