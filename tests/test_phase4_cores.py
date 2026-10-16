import copy
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from oec.phase1c import Memory, ProbeGoal, ProbePlan
from oec.phase4_single import SingleSignalSelector
from oec.phase4_endpoints import paired_points
from oec.runtime import ProtocolError, Runtime
from oec.types import Candidate, Limits

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/(name+".py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class SingleSignalTests(unittest.TestCase):
    def setup_values(self,root,mode):
        class Backend:
            def __init__(self):self.examples=0;self.states=[]
            def decide(self,state,questions,critical=False):
                self.examples+=len(questions);self.states.append(copy.deepcopy(state))
                return {key:{"type":"noul","noul":.8} for key in questions}
        backend=Backend()
        rt=Runtime(root,Limits(search_calls=1,read_calls=4,llm_calls=64,decision_calls=200,
            repair_search_reserve=0,repair_read_reserve=0,terminal_llm_reserve=6,terminal_decision_reserve=1))
        plan=ProbePlan(goals=[ProbeGoal(id="g1",description="advisor",depends_on=[])])
        memory=Memory("q",plan,unresolved=["g1"])
        candidates=[Candidate(doc_id=f"d{i}",title=f"D{i}",snippet="fact",retrieval_score=1) for i in range(10)]
        return SingleSignalSelector(backend,rt,mode),backend,memory,candidates

    def test_one_decision_per_candidate_and_zero_numerical_penalty(self):
        with tempfile.TemporaryDirectory() as raw:
            selector,backend,memory,candidates=self.setup_values(Path(raw)/"run","current")
            memory.facts=[{"text":"fact"}]
            scores=selector.score(memory,candidates)
            self.assertEqual(backend.examples,10)
            self.assertAlmostEqual(scores["d0"],.8)
            memory.hypotheses=["B"]
            selector.score(memory,candidates[1:])
            self.assertEqual(backend.examples,19)

    def test_unchanged_semantics_do_not_create_a_second_noisy_model_decision(self):
        with tempfile.TemporaryDirectory() as raw:
            selector,backend,memory,candidates=self.setup_values(Path(raw)/"run","current")
            selector.score(memory,candidates)
            memory.revision+=1
            selector.score(memory,candidates[1:])
            self.assertEqual(backend.examples,10)

    def test_initial_state_reuses_original_scores_without_new_model_calls(self):
        with tempfile.TemporaryDirectory() as raw:
            selector,backend,memory,candidates=self.setup_values(Path(raw)/"run","initial")
            before=selector.score(memory,candidates)
            memory.facts=[{"text":"new fact"}];memory.hypotheses=["B"]
            after=selector.score(memory,candidates[1:])
            self.assertEqual(after,{row.doc_id:before[row.doc_id] for row in candidates[1:]})
            self.assertEqual(backend.examples,10)
            changed=candidates[1].model_copy(update={"snippet":"changed"})
            with self.assertRaises(ProtocolError):selector.score(memory,[changed])

    def test_first_state_inputs_equal_for_both_modes(self):
        with tempfile.TemporaryDirectory() as raw:
            a,ab,am,ac=self.setup_values(Path(raw)/"a","initial")
            b,bb,bm,bc=self.setup_values(Path(raw)/"b","current")
            a.score(am,ac);b.score(bm,bc)
            self.assertEqual(ab.states,bb.states)

    def test_shared_first_scores_restore_without_recomputing_the_common_prefix(self):
        with tempfile.TemporaryDirectory() as raw:
            selector,backend,memory,candidates=self.setup_values(Path(raw)/"run","initial")
            scores={row.doc_id:.8 for row in candidates}
            selector.restore_initial(memory,candidates,scores)
            self.assertEqual(selector.score(memory,candidates[1:]),{row.doc_id:.8 for row in candidates[1:]})
            self.assertEqual(backend.examples,0)
            with self.assertRaises(ProtocolError):selector.restore_initial(memory,candidates,scores)


class EndpointTests(unittest.TestCase):
    def test_final_reader_failure_scores_zero_even_with_invalid_status(self):
        before=[{"task_id":"t","runtime_valid":False,"answer":"","status":"INVALID_RUNTIME_EVALUATION","failure_stage":"bounded_reader_output"}]
        after=[{"task_id":"t","runtime_valid":True,"answer":"B","status":"ANSWER_ATTEMPTED_UNVERIFIED"}]
        self.assertEqual(paired_points(before,after,[{"_id":"t","answer":"B"}],lambda a,b:a==b),[(0,1)])
        self.assertFalse(before[0]["runtime_valid"])
        before[0]["failure_stage"]="upstream_selection"
        with self.assertRaises(ValueError):paired_points(before,after,[{"_id":"t","answer":"B"}],lambda a,b:a==b)

    def test_duplicate_or_missing_ids_rejected(self):
        row={"task_id":"t","runtime_valid":True,"answer":"B"}
        with self.assertRaises(ValueError):paired_points([row,row],[row],[{"_id":"t","answer":"B"}],lambda a,b:a==b)
        with self.assertRaises(ValueError):paired_points([row],[row],[{"_id":"other","answer":"B"}],lambda a,b:a==b)


class TraceTests(unittest.TestCase):
    def make_run(self,root,k,sequence,skips=()):
        path=root/f"k{k}";directory=path/"task_00000";directory.mkdir(parents=True)
        manifest={"completed":True,"read_k":k,"task_ids":["t"],"phase3_policy":"utility"}
        events=[]
        for position,key in enumerate(sequence,1):
            events.extend([{"event":"response","kind":"read","response":{"id":key,"title":key,"sentences":[key]}},
                           {"event":"phase1c_selected","doc_id":key}])
            if position in skips:events.append({"event":"observation_skipped","error":"two output attempts failed"})
        (directory/"events.jsonl").write_text("".join(json.dumps(row)+"\n" for row in events))
        (directory/"result.json").write_text(json.dumps({"task_id":"t","status":"SELECTION_ONLY_NOT_EVALUATED"}))
        (path/"manifest.json").write_text(json.dumps(manifest));return path

    def test_cross_k_mismatch_counts_pairs_and_unique_questions_separately(self):
        module=load("audit_phase4_traces")
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);a=self.make_run(root,1,["a"]);b=self.make_run(root,2,["b","c"],(1,2))
            result=module.audit([a,b])
            self.assertEqual(result["prefix_mismatch_pairs"],1)
            self.assertEqual(result["prefix_mismatch_questions"],1)
            self.assertFalse(result["comparable_inputs"])
            self.assertEqual(result["observation_failures_by_run"][1]["terminal"],1)
            self.assertEqual(result["observation_failures_by_run"][1]["before_future_selection"],1)

    def test_frozen_contexts_are_nested_and_keep_full_original_sentences(self):
        module=load("audit_phase4_traces")
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);run=self.make_run(root,2,["b","a"])
            tasks=root/"tasks.jsonl";corpus=root/"corpus.jsonl"
            tasks.write_text(json.dumps({"id":"t","question":"q","corpus_ids":["a","b"]})+"\n")
            corpus.write_text("".join(json.dumps({"id":key,"title":key,"sentences":[key]})+"\n" for key in ("a","b")))
            manifest=json.loads((run/"manifest.json").read_text())
            manifest.update(tasks_sha256=module.sha(tasks),corpus_sha256=module.sha(corpus))
            (run/"manifest.json").write_text(json.dumps(manifest))
            module.freeze(run,tasks,corpus,root/"frozen")
            contexts=module.lines(root/"frozen/contexts.jsonl")
            self.assertEqual(contexts[0]["selected_sequence"],["b"])
            self.assertEqual(contexts[1]["selected_sequence"],["b","a"])
            self.assertEqual(contexts[1]["context"]["documents"][0]["sentences"][0]["text"],"b")
            self.assertEqual(contexts[0]["context"]["documents"][0],contexts[1]["context"]["documents"][0])
            self.assertEqual(set(contexts[0]["context"]),{"question","documents"})

    def test_frozen_reader_chain_uses_only_verified_prefix_contexts(self):
        module=load("audit_phase4_traces");reader=load("replay_phase4_prefix_reader")
        from oec.phase1c import ShortAnswer,Citation
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);run=self.make_run(root,2,["b","a"])
            tasks=root/"tasks.jsonl";corpus=root/"corpus.jsonl"
            tasks.write_text(json.dumps({"id":"t","question":"q","corpus_ids":["a","b"]})+"\n")
            corpus.write_text("".join(json.dumps({"id":key,"title":key,"sentences":[key]})+"\n" for key in ("a","b")))
            manifest=json.loads((run/"manifest.json").read_text());manifest.update(tasks_sha256=module.sha(tasks),corpus_sha256=module.sha(corpus))
            (run/"manifest.json").write_text(json.dumps(manifest))
            module.freeze(run,tasks,corpus,root/"frozen")
            (root/"identity.json").write_text('{"fixture":true}')
            value=ShortAnswer(answer="B",citations=[Citation(document_ref="d1",sentence_id=0)],abstain=False)
            import sys
            arguments=["reader","--contexts",str(root/"frozen"),"--tasks",str(tasks),"--corpus",str(corpus),
                       "--identity",str(root/"identity.json"),"--out",str(root/"reader")]
            with patch.object(sys,"argv",arguments),patch.object(reader.ContextReader,"read",return_value=value) as call:
                reader.main();self.assertEqual(call.call_count,2)
            output=module.lines(root/"reader/k2/predictions.jsonl")
            self.assertEqual(output[0]["supporting_facts"],[["b",0]])
            self.assertTrue(output[0]["runtime_valid"])


class SplitTests(unittest.TestCase):
    def test_exposed_ids_and_duplicate_questions_excluded_deterministically(self):
        module=load("make_phase4_unseen_split")
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);context=[[f"D{i}",[f"sentence {i}"]] for i in range(10)]
            values=[{"_id":f"t{i}","question":("already seen" if i<2 else f"question {i}"),
                     "context":context,"answer":"secret","supporting_facts":[["D0",0],["D1",0]]} for i in range(5)]
            source=root/"source.json";source.write_text(json.dumps(values))
            seen=root/"seen.jsonl";seen.write_text(json.dumps({"id":"t0"})+"\n")
            registry=root/"registry.json";registry.write_text(json.dumps({"history_complete_declared":True,
                "files":[{"path":"seen.jsonl","sha256":module.sha(seen)}]}))
            a=module.build(source,registry,root/"a",2,17)
            b=module.build(source,registry,root/"b",2,17)
            self.assertEqual(a["assigned_ids"],b["assigned_ids"])
            self.assertFalse({"t0","t1"}&set(a["assigned_ids"]))
            self.assertEqual(set(json.loads((root/"a/tasks.jsonl").read_text().splitlines()[0])),{"id","question"})
            with self.assertRaises(ValueError):module.build(source,registry,root/"too_many",4,17)

    def test_exposure_hash_change_blocks_assignment(self):
        module=load("make_phase4_unseen_split")
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);(root/"source.json").write_text('[{"_id":"t","question":"q","context":[]}]')
            (root/"seen.jsonl").write_text('{"id":"t"}\n')
            (root/"registry.json").write_text(json.dumps({"history_complete_declared":True,
                "files":[{"path":"seen.jsonl","sha256":"wrong"}]}))
            with self.assertRaises(ValueError):module.build(root/"source.json",root/"registry.json",root/"out",1,17)


if __name__=="__main__":unittest.main()
