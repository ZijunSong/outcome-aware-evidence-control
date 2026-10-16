"""Gold-free frozen-candidate loading. Does not retrieve or inspect supporting facts."""
import hashlib
import json
from pathlib import Path
from .runtime import ProtocolError, digest
from .types import Candidate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(pool_root):
    manifest = json.loads((Path(pool_root) / "manifest.json").read_text())
    if manifest.get("gold_used") is not False:
        raise ProtocolError("Refusing a pool that used gold")
    if "pool_files" not in manifest or "pool_files_sha256" not in manifest:
        raise ProtocolError("Pool manifest is missing file hashes")
    if digest(manifest["pool_files"]) != manifest["pool_files_sha256"]:
        raise ProtocolError("Pool file-hash ledger does not match")
    return manifest


def load_task_pool(pool_root, task, manifest=None):
    pool_root = Path(pool_root)
    manifest = manifest or load_manifest(pool_root)
    path = pool_root / "pools" / f"{task.id}.json"
    if not path.is_file():
        raise ProtocolError("Missing frozen pool for this question")
    actual = sha(path)
    if manifest["pool_files"].get(task.id) != actual:
        raise ProtocolError("Frozen pool file hash changed")
    pool = json.loads(path.read_text())
    if pool.get("task_id") != task.id or pool.get("question") != task.question:
        raise ProtocolError("Frozen pool question does not match the assigned task")
    rows = pool["candidates"]
    ids = [row["doc_id"] for row in rows]
    if not ids or len(ids) != len(set(ids)) or ids != list(task.corpus_ids):
        raise ProtocolError("Task scope is not the frozen pool")
    if [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ProtocolError("Frozen ranks are not contiguous")
    return pool, rows


def candidates_from_pool(retriever, task, pool_root, manifest=None, snippet_chars=240):
    pool, rows = load_task_pool(pool_root, task, manifest)
    output = []
    for row in rows:
        document = retriever.documents[row["doc_id"]]
        snippet = " ".join(document.sentences)[:snippet_chars]
        content = hashlib.sha256((document.title + "\n" + "\n".join(document.sentences)).encode()).hexdigest()
        if snippet != row["snippet"] or document.title != row["title"]:
            raise ProtocolError("Frozen snippet or content version does not match the corpus")
        if row.get("content_sha256") not in (content, row["doc_id"]):
            raise ProtocolError("Frozen content hash does not match the corpus text")
        output.append(Candidate(doc_id=row["doc_id"], title=row["title"], snippet=snippet,
                                retrieval_score=row["retrieval_score"]))
    return output


def install(pool_root, module):
    original = module.fixed_candidates
    manifest = load_manifest(pool_root)

    def frozen(retriever, task, snippet_chars=240):
        return candidates_from_pool(retriever, task, pool_root, manifest, snippet_chars)

    module.fixed_candidates = frozen
    return original, manifest
