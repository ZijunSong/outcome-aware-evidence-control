"""Read-only trace audit; optionally freeze nested reader contexts from ONE max-K path."""
import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.runtime import digest


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique(rows, key):
    values = {row[key]: row for row in rows}
    if not rows or len(values) != len(rows):
        raise ValueError("Empty or duplicate ids")
    return values


def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(run):
    manifest = json.loads((run/"manifest.json").read_text())
    if not manifest.get("completed"):
        raise ValueError("Incomplete run")
    rows, skipped = {}, Counter()
    for directory in sorted(run.glob("task_*")):
        result = json.loads((directory/"result.json").read_text())
        key = result["task_id"]
        if key in rows or result["status"] != "SELECTION_ONLY_NOT_EVALUATED":
            raise ValueError("Duplicate/invalid deferred result")
        events = lines(directory/"events.jsonl")
        if any(row["event"] == "invalid_run" for row in events):
            raise ValueError("Upstream selection failed")
        chosen = [row["doc_id"] for row in events if row["event"] == "phase1c_selected"]
        reads = [row["response"] for row in events if row["event"] == "response" and row.get("kind") == "read"]
        if len(chosen) != manifest["read_k"] or len(set(chosen)) != len(chosen):
            raise ValueError("Incorrect read sequence length")
        if [row["id"] for row in reads] != chosen:
            raise ValueError("Read responses do not match selected order")
        position, failures = 0, []
        for event in events:
            if event["event"] == "phase1c_selected":
                position += 1
            if event["event"] == "observation_skipped":
                if not 1 <= position <= len(chosen):
                    raise ValueError("Unpositioned observation failure")
                failures.append({"position": position, "error": event.get("error", "")})
                skipped[str(position)] += 1
        rows[key] = {"sequence": chosen, "reads": reads, "observation_failures": failures,
                     "events_sha256": sha(directory/"events.jsonl")}
    if set(rows) != set(manifest["task_ids"]) or len(rows) != len(manifest["task_ids"]):
        raise ValueError("Task manifest mismatch")
    return manifest, rows, dict(skipped)


def audit(runs):
    loaded = [(path, *read_run(path)) for path in runs]
    if len({manifest["read_k"] for _,manifest,_,_ in loaded}) != len(loaded):
        raise ValueError("Use one run per K for one policy")
    loaded.sort(key=lambda row: row[1]["read_k"])
    fields = ("tasks_sha256", "corpus_sha256", "plans_sha256", "llm_protocol", "source_sha256",
              "additional_source_sha256", "phase3_policy", "phase3_model_identity_sha256", "observer_ledger")
    differences = {key: [str(path) for path,manifest,_,_ in loaded if manifest.get(key) != loaded[0][1].get(key)]
                   for key in fields}
    differences = {key:value for key,value in differences.items() if value}
    missing = {str(path):[key for key in fields if key not in manifest]
               for path,manifest,_,_ in loaded}
    missing = {key:value for key,value in missing.items() if value}
    mismatches = []
    for (_,a,ar,_),(_,b,br,_) in itertools.combinations(loaded,2):
        if set(ar) != set(br):
            raise ValueError("Cross-K task sets differ")
        for key in sorted(ar):
            if ar[key]["sequence"] != br[key]["sequence"][:a["read_k"]]:
                mismatches.append({"task_id":key,"shorter_k":a["read_k"],"longer_k":b["read_k"]})
    return {"protocol":"phase4-trace-audit-v1", "comparable_inputs":not differences and not missing,
            "missing_manifest_fields":missing,
            "manifest_differences":differences, "prefix_mismatch_pairs":len(mismatches),
            "prefix_mismatch_questions":len({row["task_id"] for row in mismatches}),
            "mismatches":mismatches,
            "observation_failures_by_run":[{"run":str(path),"k":manifest["read_k"],"by_position":skipped,
                "terminal":sum(len([x for x in row["observation_failures"] if x["position"]==manifest["read_k"]]) for row in rows.values()),
                "before_future_selection":sum(len([x for x in row["observation_failures"] if x["position"]<manifest["read_k"]]) for row in rows.values())}
                for path,manifest,rows,skipped in loaded]}


def freeze(run, tasks_path, corpus_path, out, order="read_prefix"):
    if order not in ("read_prefix","canonical"):raise ValueError("Unknown document order")
    manifest, rows, skipped = read_run(run)
    tasks, corpus = unique(lines(tasks_path),"id"), unique(lines(corpus_path),"id")
    if sha(tasks_path)!=manifest["tasks_sha256"] or sha(corpus_path)!=manifest["corpus_sha256"]:
        raise ValueError("Input hash mismatch")
    if set(rows)!=set(tasks):
        raise ValueError("Assigned tasks differ")
    bundles=[]
    for key,row in sorted(rows.items()):
        if not set(row["sequence"])<=set(tasks[key]["corpus_ids"]):
            raise ValueError("Selected document outside frozen public scope")
        for value in row["reads"]:
            original=corpus[value["id"]]
            if value["title"]!=original["title"] or value["sentences"]!=original["sentences"]:
                raise ValueError("Read source text changed")
        for k in range(1,manifest["read_k"]+1):
            sequence=row["sequence"][:k]
            documents=[corpus[doc_id] for doc_id in (sequence if order=="read_prefix" else sorted(sequence))]
            context={"question":tasks[key]["question"],"documents":[{"document_ref":f"d{i+1}",
                "title":doc["title"],"sentences":[{"sentence_id":sid,"text":text}
                for sid,text in enumerate(doc["sentences"])]} for i,doc in enumerate(documents)]}
            bundles.append({"task_id":key,"k":k,"selected_sequence":sequence,
                "context":context,"context_hash":digest(context),"source_events_sha256":row["events_sha256"]})
    out.mkdir(parents=True,exist_ok=False)
    (out/"contexts.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in bundles))
    header={"protocol":"phase4-frozen-prefix-contexts-v1","source_run":str(run.resolve()),
        "policy":manifest.get("phase3_policy"),"task_ids":sorted(tasks),"max_k":manifest["read_k"],
        "tasks_sha256":sha(tasks_path),"corpus_sha256":sha(corpus_path),
        "source_manifest_sha256":sha(run/"manifest.json"),"contexts_sha256":sha(out/"contexts.jsonl"),
        "reader_order_policy":order,
        "online_gold_access":False,"warning":"Frozen contexts only; no model calls or new efficacy result."}
    (out/"manifest.json").write_text(json.dumps(header,ensure_ascii=False,indent=2))
    return header


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    inspect=sub.add_parser("audit");inspect.add_argument("--runs",nargs="+",type=Path,required=True);inspect.add_argument("--out",type=Path,required=True)
    build=sub.add_parser("freeze")
    for name in ("run","tasks","corpus","out"):build.add_argument("--"+name,type=Path,required=True)
    build.add_argument("--order",choices=("read_prefix","canonical"),default="read_prefix")
    args=parser.parse_args()
    if args.command=="freeze":
        report=freeze(args.run,args.tasks,args.corpus,args.out,args.order)
    else:
        report=audit(args.runs);args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x") as handle:json.dump(report,handle,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
