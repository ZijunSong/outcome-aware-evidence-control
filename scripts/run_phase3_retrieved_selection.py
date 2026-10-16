"""Select from a frozen BM25 pool. The candidate list is not retrieved again."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oec.phase1c import fixed_candidates
from oec.runtime import ProtocolError
from oec.types import Candidate
import oec.phase1c as phase1c


def install_frozen_pool(pool_root):
    original = fixed_candidates

    def frozen(retriever, task, snippet_chars=240):
        path = Path(pool_root) / "pools" / f"{task.id}.json"
        pool = json.loads(path.read_text())
        rows = pool["candidates"]
        if [row["doc_id"] for row in rows] != list(task.corpus_ids):
            raise ProtocolError("Task scope is not the frozen pool")
        candidates = []
        for row in rows:
            document = retriever.documents[row["doc_id"]]
            snippet = " ".join(document.sentences)[:snippet_chars]
            if snippet != row["snippet"] or document.title != row["title"]:
                raise ProtocolError("Frozen snippet does not match the corpus text")
            candidates.append(Candidate(doc_id=row["doc_id"], title=row["title"], snippet=snippet,
                                        retrieval_score=row["retrieval_score"]))
        return candidates

    phase1c.fixed_candidates = frozen
    return original


def main():
    if "--pool" not in sys.argv:
        raise SystemExit("Pass --pool DIR before the phase3 arguments")
    index = sys.argv.index("--pool")
    pool = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    install_frozen_pool(pool)
    from scripts.run_phase3_selection import main as run
    run()


if __name__ == "__main__":
    main()
