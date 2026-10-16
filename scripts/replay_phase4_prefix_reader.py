"""Gold-free reader diagnostic over verified nested contexts. No new selection or repair."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError
from oec.reader_v2 import ContextReader
from oec.runtime import Runtime, ProtocolError, HTTPFailure, BudgetExceeded, digest
from oec.types import Limits
from scripts.audit_phase4_traces import lines, unique, sha


def main():
    parser=argparse.ArgumentParser()
    for name in ("contexts","tasks","corpus","identity","out"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    header=json.loads((args.contexts/"manifest.json").read_text())
    if header.get("protocol")!="phase4-frozen-prefix-contexts-v1" or header.get("online_gold_access") is not False:
        raise ValueError("Unexpected frozen-context profile")
    for name,path in (("tasks",args.tasks),("corpus",args.corpus),("contexts",args.contexts/"contexts.jsonl")):
        if header[name+"_sha256"]!=sha(path):raise ValueError("Frozen input hash changed")
    identity=json.loads(args.identity.read_text())
    if not isinstance(identity,dict) or not identity:raise ValueError("Missing model identity")
    tasks,corpus=unique(lines(args.tasks),"id"),unique(lines(args.corpus),"id")
    if sorted(tasks)!=header["task_ids"]:raise ValueError("Assigned task ids differ")
    bundles=lines(args.contexts/"contexts.jsonl");seen=set();sequences={}
    for row in bundles:
        key=(row["task_id"],row["k"])
        if key in seen or key[0] not in tasks:raise ValueError("Duplicate/unknown task-K pair")
        seen.add(key)
        sequence=row["selected_sequence"]
        if len(sequence)!=row["k"] or len(set(sequence))!=len(sequence) or not set(sequence)<=set(tasks[key[0]]["corpus_ids"]):
            raise ValueError("Illegal prefix source ids")
        ordered=sequence if header["reader_order_policy"]=="read_prefix" else sorted(sequence)
        expected={"question":tasks[key[0]]["question"],"documents":[{"document_ref":f"d{i+1}",
            "title":corpus[doc_id]["title"],"sentences":[{"sentence_id":sid,"text":text}
            for sid,text in enumerate(corpus[doc_id]["sentences"])]} for i,doc_id in enumerate(ordered)]}
        if row["context"]!=expected or row["context_hash"]!=digest(expected):raise ValueError("Reader source text differs")
        sequences[key]=sequence
    if seen!={(key,k) for key in tasks for k in range(1,header["max_k"]+1)}:raise ValueError("Incomplete task-K grid")
    for key in tasks:
        longest=sequences[key,header["max_k"]]
        if any(sequences[key,k]!=longest[:k] for k in range(1,header["max_k"]+1)):
            raise ValueError("Contexts are not nested prefixes of one path")
    base=os.environ.get("OEC_LLM_BASE_URL","http://127.0.0.1:8000/v1")
    model=os.environ.get("OEC_LLM_MODEL","Qwen3-14B")
    if identity.get("llm_served_model",model)!=model:raise ValueError("Model name and identity differ")
    policy=ContextReader(None,base,model,sha(args.identity)).signature()|{
        "reader_source_sha256":sha(Path(__file__).resolve().parents[1]/"oec/reader_v2.py"),
        "diagnostic_order":header["reader_order_policy"]}
    pool={digest({"context":row["context"],"reader_policy":policy}):row["context"] for row in bundles}
    args.out.mkdir(parents=True,exist_ok=False)
    limits=Limits(search_calls=1,read_calls=1,llm_calls=8,decision_calls=5,
        repair_search_reserve=0,repair_read_reserve=0,terminal_llm_reserve=6,terminal_decision_reserve=1)
    entries={}
    for key in sorted(pool):
        rt=Runtime(args.out/"reader_pool"/key,limits)
        reader=ContextReader(rt,base,model,sha(args.identity),os.environ.get("OEC_LLM_API_KEY","local"))
        try:
            answer=reader.read(pool[key]);entry={"runtime_valid":True,"answer":answer.model_dump(),"error":None}
        except HTTPFailure as exc:
            if exc.status in (400,401,403,404,422):raise RuntimeError("Reader configuration failure") from exc
            entry={"runtime_valid":False,"answer":None,"error":str(exc)}
        except (ProtocolError,OSError,BudgetExceeded,ValidationError) as exc:
            entry={"runtime_valid":False,"answer":None,"error":str(exc)}
        entry.update(key=key,context=pool[key],reader_policy=policy,usage=dict(rt.used)|dict(rt.tokens))
        (rt.path.parent/"entry.json").write_text(json.dumps(entry,ensure_ascii=False,indent=2));entries[key]=entry
    ledger_hash=digest({key:digest(value) for key,value in sorted(entries.items())})
    by_k={k:[] for k in range(1,header["max_k"]+1)}
    for row in bundles:
        key=digest({"context":row["context"],"reader_policy":policy});entry=entries[key]
        answer=entry["answer"];valid=entry["runtime_valid"]
        refs={doc["document_ref"]:doc for doc in row["context"]["documents"]}
        by_k[row["k"]].append({"task_id":row["task_id"],"runtime_valid":valid,
            "answer":answer["answer"] if valid else "", "verified":False,
            "supporting_facts":sorted({(refs[c["document_ref"]]["title"],c["sentence_id"]) for c in answer["citations"]}) if valid else [],
            "status":("ANSWER_ATTEMPTED_ABSTAINED" if answer["abstain"] else "ANSWER_ATTEMPTED_UNVERIFIED") if valid else "INVALID_RUNTIME_EVALUATION",
            "failure_stage":None if valid else "bounded_reader_output", "selected_doc_ids":sorted(row["selected_sequence"]),
            "selected_titles":[doc["title"] for doc in row["context"]["documents"]],"shared_reader_key":key,
            "reader_ledger_sha256":ledger_hash,"k":row["k"],"usage":entry["usage"]})
    for k,rows in by_k.items():
        directory=args.out/f"k{k}";directory.mkdir()
        (directory/"predictions.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows))
    report={"protocol":"phase4-frozen-prefix-reader-v1","reader_ledger_sha256":ledger_hash,
        "reader_policy":policy,"source_header":header,"n":len(tasks),"unique_contexts":len(pool),
        "bounded_reader_failures":sum(not row["runtime_valid"] for rows in by_k.values() for row in rows)}
    (args.out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report["bounded_reader_failures"]:raise SystemExit(2)


if __name__=="__main__":main()
