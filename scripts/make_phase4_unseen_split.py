"""Offline split preparation. Exposure completeness is an owner declaration, not inferred."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    text=path.read_text()
    value=json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value,list) or not value:raise ValueError("Exposure file must contain nonempty rows")
    return value


def norm(question):
    return " ".join(question.casefold().split())


def build(source, registry_path, out, n, seed):
    if n<1:raise ValueError("Sample size must be positive")
    source_rows=json.loads(source.read_text())
    by_id={row["_id"]:row for row in source_rows}
    if not source_rows or len(by_id)!=len(source_rows):raise ValueError("Empty/duplicate source ids")
    registry=json.loads(registry_path.read_text())
    if registry.get("history_complete_declared") is not True or not registry.get("files"):
        raise ValueError("Supply the declared exposure file registry; do not infer completeness from 700 ids")
    used,records=set(),[]
    for record in registry["files"]:
        path=Path(record["path"])
        if not path.is_absolute():path=registry_path.parent/path
        actual=sha(path)
        if actual!=record["sha256"]:raise ValueError("Exposure file hash changed")
        values=rows(path)
        ids=[row.get("id",row.get("task_id",row.get("_id"))) for row in values]
        if any(not isinstance(key,str) or not key for key in ids) or len(set(ids))!=len(ids):
            raise ValueError("Malformed or duplicate exposure ids")
        used.update(ids)
        records.append({"path":str(path.resolve()),"sha256":actual,"n":len(ids)})
    used_questions={norm(by_id[key]["question"]) for key in used if key in by_id}
    candidates=[];ineligible=0
    for row in source_rows:
        context=row["context"]
        if len(context)!=10 or len({title for title,_ in context})!=10 or not all(
            sentences and all(isinstance(text,str) for text in sentences) for _,sentences in context):
            ineligible+=1;continue
        if row["_id"] not in used and norm(row["question"]) not in used_questions:candidates.append(row)
    candidates.sort(key=lambda row:hashlib.sha256(f"{seed}:{row['_id']}".encode()).hexdigest())
    selected,question_seen=[],set(used_questions)
    for row in candidates:
        if norm(row["question"]) not in question_seen:
            selected.append(row);question_seen.add(norm(row["question"]))
    available=len(selected)
    if available<n:raise ValueError(f"Only {available} unexposed eligible questions according to the supplied registry")
    selected=selected[:n]
    out.mkdir(parents=True,exist_ok=False)
    tasks=out/"tasks.jsonl";gold=out/"gold_OFFLINE.json"
    tasks.write_text("".join(json.dumps({"id":row["_id"],"question":row["question"]},ensure_ascii=False)+"\n" for row in selected))
    gold.write_text(json.dumps(selected,ensure_ascii=False))
    manifest={"protocol":"phase4-assigned-unseen-v1","status":"ASSIGNED_NOT_SCORED",
        "n":n,"seed":seed,"source_sha256":sha(source),"registry_sha256":sha(registry_path),
        "exposure_files":records,"exposed_id_count":len(used),"source_ineligible":ineligible,
        "available_before_assignment":available,"assigned_ids":[row["_id"] for row in selected],
        "task_ids_sha256":hashlib.sha256(json.dumps(sorted(row["_id"] for row in selected),separators=(",",":")).encode()).hexdigest(),
        "tasks_sha256":sha(tasks),"gold_OFFLINE_sha256":sha(gold),
        "online_fields":["id","question"],
        "scope":"Unseen relative to supplied recorded ids and exact normalized duplicate questions. "
                "Completeness of historical records is declared by the owner; unrecorded exposure cannot be certified."}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser()
    for name in ("source","registry","out"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--n",type=int,required=True);parser.add_argument("--seed",type=int,default=20261016)
    args=parser.parse_args();result=build(args.source,args.registry,args.out,args.n,args.seed)
    print(json.dumps({key:result[key] for key in ("n","available_before_assignment","exposed_id_count","status")},indent=2))


if __name__=="__main__":main()
