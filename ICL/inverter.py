from __future__ import annotations

import json
import numpy as np
from ICL.clients import chat_llm
from ICL.data import (
    EvaluatedCandidate, InversionError, InversionResult, IterationRecord, ConstraintModel
)
from ICL.encoding import ModelCodec
from ICL.misfit import evaluate_misfit
from ICL.prompt import build_initial_prompt, build_refinement_prompt, build_joint_prompt

class AEMICLInverter:
    def __init__(self, forward_operator, config) -> None:
        self.config = config
        self.forward_operator = forward_operator
        self.model_codec = ModelCodec(config.rho_min, config.rho_max)

    def _call_and_parse(self, prompt):
        result = chat_llm(prompt)
        result_dic = json.loads(result)
        model_code = np.asarray(result_dic["rho_code"])
        return model_code

    def _evaluate(self, model_code, query):
        model_rhos = self.model_codec.decode_log(model_code)
        predicted = self.forward_operator(
            model_rhos, self.config.layer_thicknesses,
            query.height, query.times,
        )
        misfit = evaluate_misfit(
            query.response, predicted, relative_error=self.config.relative_error,
            noise_floor=self.config.noise_floor, data_std=query.data_std,
        )
        return EvaluatedCandidate(model_code, model_rhos, predicted, misfit)

    def invert(self, query, examples, response_codec):
        prompt = build_initial_prompt(query, examples, self.config, response_codec)
        model_code = self._call_and_parse(prompt)
        return model_code

    def loop_invert(self, query, examples, response_codec):
        prompt = build_initial_prompt(query, examples, self.config, response_codec)
        history = []
        best = None
        stale_rounds = 0
        stop_reason = "max_iterations"
        iteration = 0
        while iteration < self.config.max_iterations + 1:
            try:
                record = IterationRecord(iteration=iteration)
                record.prompt = prompt
                model_code = self._call_and_parse(prompt)
                candidate= self._evaluate(model_code, query)
                record.candidate = candidate
                improvement = float("inf") if best is None else best.rms - candidate.rms
                record.improved = best is None or candidate.rms < best.rms
                print(f"iteration={record.iteration}, RMS={record.candidate.rms:.6f}, "
                      f"improved={record.improved}")

                history.append(record)

                if record.improved:
                    best = candidate
                stale_rounds = 0 if improvement > self.config.min_improvement else stale_rounds + 1
                if best.rms <= self.config.target_rms:
                    stop_reason = "target_rms"
                    break
                if self.config.patience is not None and stale_rounds >= self.config.patience:
                    stop_reason = "stagnation"
                    break
                if iteration < self.config.max_iterations:
                    prompt = build_refinement_prompt(query, examples, self.config, response_codec,best,history)
            except Exception as e:
                print(f"run={iteration} 出错，从当前序号重试: {type(e).__name__}: {e}")
                continue
            
            iteration += 1

        if best is None:
            raise InversionError(
                f"No valid initial model ({stop_reason}): {history[-1].error}", history)
        return InversionResult(query.sounding_id, float(query.height), best, history, stop_reason)

    def joint_invert(self, query, examples, response_codec, constraint_model):
        prompt = build_joint_prompt(query, examples, self.config, response_codec, constraint_model)
        model_code = self._call_and_parse(prompt)
        return model_code
