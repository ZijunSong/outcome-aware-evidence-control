import json
import os
from pathlib import Path
from .jev import validate_answers
from .runtime import ProtocolError, digest


class LocalSearchJev:
    """Native batched predictor adapter. Pin the upstream revision: _forward is a private hook."""
    def __init__(self, runtime, model_run):
        from training.inference import JEVPredictor
        from training.schema import jev_row_to_examples
        if not model_run:
            raise ProtocolError("--local-model-run must identify a trained or released model")
        self.runtime = runtime
        self.convert = jev_row_to_examples
        base_model = os.environ.get("OEC_JEV_BASE_MODEL", "").strip()
        local_base = Path("/data/ppnm/models/Qwen3.5-4B")
        if not base_model and (local_base / "config.json").is_file() and any(local_base.glob("model.safetensors*")):
            base_model = str(local_base)
        self.predictor = JEVPredictor(
            model_run, base_model=base_model or None,
            device=os.environ.get("OEC_JEV_DEVICE", "cuda"),
            batch_size=int(os.environ.get("OEC_JEV_BATCH_SIZE", "16")), merge_lora=True)
        self.critical = False
        original_forward = self.predictor._forward

        def metered_forward(encoded):
            metadata = {"examples": len(encoded),
                        "lengths": [len(row["input_ids"]) for row in encoded],
                        "encoded_batch_hash": digest([row["input_ids"] for row in encoded])}
            return self.runtime.invoke("decision", metadata, lambda: original_forward(encoded),
                                       self.critical, "local_searchjev")
        self.predictor._forward = metered_forward

    @staticmethod
    def targeted_state(state, question_id, question):
        view = dict(state)
        instruction = question["instructions"]
        doc_id = instruction.get("candidate_doc_id") if isinstance(instruction, dict) else None
        if doc_id is not None:
            view["candidates"] = [row for row in state.get("candidates", []) if row["doc_id"] == doc_id]
        if "claim_rows" in state:
            view["claim_rows"] = [row for row in state["claim_rows"] if row["claim"]["id"] == question_id]
        return view

    def decide(self, state, questions, critical=False):
        self.critical = critical
        payload = {"state": state, "questions": questions}
        self.runtime.event("local_typed_request", request=payload, request_hash=digest(payload))
        examples, keys = [], []
        for question_id, question in questions.items():
            native_question = dict(question)
            if not isinstance(question["instructions"], str):
                native_question["instructions"] = json.dumps(question["instructions"], ensure_ascii=False)
            row = {"sample_id": "oec-" + question_id,
                   "state": self.targeted_state(state, question_id, question),
                   "questions": {question_id: native_question}}
            converted, dropped = self.convert(row, label_space=self.predictor.label_space, require_answer=False)
            if len(converted) != 1 or any(dropped.values()):
                raise ProtocolError("SearchJev schema conversion dropped a requested decision")
            examples.extend(converted)
            keys.append(question_id)
        results = self.predictor.decide_batch(examples)
        if len(results) != len(keys):
            raise ProtocolError("SearchJev batch result count mismatch")
        self.runtime.event("local_typed_response", raw_results={key: result.to_dict()
                           for key, result in zip(keys, results)})
        if any(result.truncated for result in results):
            raise ProtocolError("SearchJev truncated decision context; shorten the explicit state before rerunning")
        answers = {}
        for key, result in zip(keys, results):
            question = questions[key]
            if question["type"] == "noul":
                if set(result.probabilities) != {"true", "false"}:
                    raise ProtocolError("Unexpected native Noul option labels")
                answers[key] = {"type": "noul", "noul": result.probabilities["true"]}
            else:
                probabilities = result.probabilities
                size = len(probabilities)
                # Native SearchJev confidence is max(p); normalize only the routing statistic.
                confidence = ((max(probabilities.values()) - 1/size) / (1 - 1/size)) if size > 1 else 1
                answers[key] = {"type": "choice", "choice": result.choice,
                                "probabilities": probabilities, "confidence": confidence}
        return validate_answers(payload, {"answers": answers})
