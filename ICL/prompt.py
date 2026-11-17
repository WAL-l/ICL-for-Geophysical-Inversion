from __future__ import annotations
import numpy as np
import json
import math
import operator

def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def layer_depths(thicknesses, n_layers=30):
    thicknesses = np.asarray(thicknesses, dtype=float).reshape(-1)
    if thicknesses.size != n_layers - 1:
        raise ValueError(f"expected {n_layers - 1} fixed layer thicknesses")
    tops = np.r_[0.0, np.cumsum(thicknesses)]
    bottoms = np.r_[np.cumsum(thicknesses), np.inf]
    return tops, bottoms


def build_initial_prompt(query, examples, config, response_codec):
    tops, bottoms = layer_depths(config.layer_thicknesses, config.n_layer)
    layer_grid = [
        {
            "layer": i + 1,
            "top_m": round(float(tops[i]), 3),
            "bottom_m": None if not np.isfinite(bottoms[i]) else round(float(bottoms[i]), 3),
        }
        for i in range(config.n_layer)
    ]
    demos = []
    for rank, ex in enumerate(examples, start=1):
        demos.append(
            {
                "id": rank,
                "height_m": round(ex.height, 3),
                "dBdt_code": ex.response_code,
                "time": ex.time.tolist(),
                "rho_code": ex.model_code.tolist(),
            }
        )

    lines = ["You are an expert geophysicist in airborne time-domain electromagnetic (AEM) inversion.",
             f"Your task is to infer physically plausible {config.n_layer}-layer rho_code for the query sounding using the supplied examples as local empirical evidence.",
             "Do not assume that the available data uniquely determine every layer.",
             "",
             "(1). Fixed acquisition configuration",
             "The transmitter waveform, loop geometry, receiver offset, response scaling, time gates, and layer thicknesses are fixed and identical for all examples and the query.",
             "Flight height may differ and is a known acquisition condition.",
             f"Each model contains {config.n_layer} values ordered from the shallowest layer to the deepest layer.",
             "rho_code definition:",
             "rho_code is a logarithmic resistivity representation:",
             "rho_code = log10(rho)",
             "dBdt_code definition:",
             "The dBdt_code is encoded from the response data using a log-10 min-max normalization process",
             r"dBdt\_code = round(clamp( \frac{\log_{10}(response) - log\_min}{log\_max - log\_min}, 0.0, 1.0) \times dBdt\_code\_max)",
             "The dBdt_code is a index-aligned representation, do not interpret them as time, depth, height, or resistivity.",
             "",
             "(2). Model parameterization:",
             f"The physical resistivity range is [{config.rho_min}, {config.rho_max}] Ohm-m."
             f"The corresponding model range is [{np.log10(config.rho_min)}, {np.log10(config.rho_max)}]."
             "Therefore:",
             f"rho_code = {np.log10(config.rho_min)} corresponds to {config.rho_min} Ohm-m",
             f"rho_code = {np.log10(config.rho_max)} corresponds to {config.rho_max} Ohm-m",
             "Every rho_code output value must be a finite number in rho_code = log10(rho) space, not physical Ohm-m.",
             "The Layer grid: " + _compact_json(layer_grid),
             "",
             "(3). How to use the examples",
             "Use every example as a local input-output relationship between:",
             "1. flight height",
             "2. the complete aligned response curve (dBdt_code),",
             f"3. and the corresponding {config.n_layer}-layer rho_code.",
             "Account for differences in flight height. Do not explain a response difference as a subsurface resistivity change when it can plausibly be associated with a height difference.",
             "Do not copy a example model unchanged unless both its response and flight height are effectively identical to the query.",
             "",
             "(4). Physics-guided inference rules",
             "1. Treat early gates as predominantly sensitive to shallower structure and later gates as increasingly sensitive to deeper structure, while remembering that AEM sensitivity is broad and there is no one-to-one gate-to-layer mapping.",
             "2. Favor coherent changes across neighboring layers over independent layer-by-layer edits.",
             "3. Prefer a smooth, minimum-structure profile unless the example and decay shape consistently support an interface or conductive/resistive body.",
             "4. Preserve supported conductive or resistive structures, smoothness does not require a monotonic profile."
             "5. Avoid unsupported one-layer spikes, rapid alternating high-low oscillations, and extreme deep-layer structure that is not constrained by the late-time response.",
             # "6. Do not copy an example merely because it is the closest available example. Equally, do not force the output to differ from an example solely to satisfy a non-copying rule. Any change should have an evidential basis."
             "",
             "--- EXAMPLES ---", ]

    for i, ex in enumerate(demos):
        lines.append(f"Example {i} height: {ex['height_m']}")
        toks_d = ex["dBdt_code"]
        dstr = ",".join(str(x) for x in toks_d)
        lines.append(f"Example {i} dBdt_code: [{dstr}]")
        rhos = ex["rho_code"]
        rstr = ",".join(f"{x:.2f}" for x in rhos)
        lines.append(f"Example {i} rho_code: [{rstr}]")
        lines.append("")

    lines.extend([
        "--- QUERY CURVE ---",
    ])
    lines.append(f"Query height: {round(query.height, 3)}")
    dstr = ",".join(str(x) for x in response_codec.encode_row_norm(query.response))
    lines.append(f"Query dBdt_code: [{dstr}]")

    lines.extend([
        "",
        "--- OUTPUT FORMAT ---",
        f"Return exactly {config.n_layer} log10(rho) values in [{np.log10(config.rho_min)}, {np.log10(config.rho_max)}].",
        "Provide your inferred rho_code as JSON in this exact format:",
        '{"rho_code": [rho1, rho2, rho3, ..., rhoN]}   // N equals number of layers',
        "",
        "Generate ONLY the JSON. No extra explanation text.",

    ])

    return "\n".join(lines)


def build_refinement_prompt(query, examples, config, response_codec,
                            best,
                            history
                            ) -> str:
    tops, bottoms = layer_depths(config.layer_thicknesses, config.n_layer)
    layer_grid = [
        {
            "layer": i + 1,
            "top_m": round(float(tops[i]), 3),
            "bottom_m": None if not np.isfinite(bottoms[i]) else round(float(bottoms[i]), 3),
        }
        for i in range(config.n_layer)
    ]
    demos = []
    for rank, ex in enumerate(examples, start=1):
        demos.append(
            {
                "id": rank,
                "height_m": round(ex.height, 3),
                "dBdt_code": ex.response_code,
                "time": ex.time.tolist(),
                "rho_code": ex.model_code.tolist(),
            }
        )

    lines = [
        "You are an expert geophysicist performing iterative airborne time-domain electromagnetic (AEM) inversion.",
        "You work inside an optimization loop:",
        "1. You propose a resistivity model.",
        "2. An external forward solver evaluates that model.",
        "3. The external program returns the predicted response and the measured misfit.",
        "4. You use this feedback to propose the next candidate.",
        "Your objective is to find a physically plausible model with the lowest externally evaluated misfit within the available evaluation budget.",
        "Do not assume a unique solution or claim that a global minimum has been reached.",
        "",
        "(1). Fixed acquisition configuration",
        "The transmitter waveform, loop geometry, receiver offset, response scaling, time gates, and layer thicknesses are fixed and identical for all examples and the query.",
        "Flight height may differ and is a known acquisition condition.",
        f"Each model contains {config.n_layer} values ordered from the shallowest layer to the deepest layer.",
        "rho_code definition:",
        "rho_code is a logarithmic resistivity representation:",
        "rho_code = log10(rho)",
        "dBdt_code definition:",
        "The dBdt_code is encoded from the response data using a log-10 min-max normalization process",
        r"dBdt\_code = round(clamp( \frac{\log_{10}(response) - log\_min}{log\_max - log\_min}, 0.0, 1.0) \times dBdt\_code\_max)",
        "The dBdt_code is a index-aligned representation, do not interpret them as time, depth, height, or resistivity.",
        "",
        "(2). Model parameterization:",
        f"The physical resistivity range is [{config.rho_min}, {config.rho_max}] Ohm-m."
        f"The corresponding model range is [{np.log10(config.rho_min)}, {np.log10(config.rho_max)}]."
        "Therefore:",
        f"rho_code = {np.log10(config.rho_min)} corresponds to {config.rho_min} Ohm-m",
        f"rho_code = {np.log10(config.rho_max)} corresponds to {config.rho_max} Ohm-m",
        "Every rho_code output value must be a finite number in rho_code = log10(rho) space, not physical Ohm-m.",
        "The Layer grid: " + _compact_json(layer_grid),
        "",
        "(3). How to use the examples",
        "The example illustrates the relationship between the actual model and the response.",
        "Use every example as a local input-output relationship between:",
        "1. flight height",
        "2. the complete aligned response curve (dBdt_code),",
        f"3. and the corresponding {config.n_layer}-layer rho_code.",
        "Account for differences in flight height. Do not explain a response difference as a subsurface resistivity change when it can plausibly be associated with a height difference.",
        "Do not copy a example model unchanged unless both its response and flight height are effectively identical to the query.",
        "",
        "(3). Use the evaluated history",
        "Study both successful and unsuccessful trials:",
        "Which model parameters changed?",
        "How large was the actual change in log10-resistivity?",
        "How did the complete predicted response change?",
        "Which residual patterns improved or worsened?",
        "Did the total reported misfit decrease?",
        "Treat differences between evaluated models as directional evidence along the actual tested perturbations.",
        "Local response-model relationships may vary with the model and step size. Do not extrapolate a successful direction indefinitely.",
        "",
        "(4). Construct the next candidate",
        "Modify the given BEST MODEL to generate the next candidate.",
        "",
        "Choose the proposal using the available evidence:",
        # "A. A useful direction has been demonstrated:",
        # "Continue or refine that direction cautiously, respecting the step limit and the full residual pattern.",
        # "B. A larger step performed worse than a smaller step in the same direction:",
        # "Consider an intermediate or smaller step from the appropriate evaluated reference. Do not assume the larger step is better because it changes more parameters.",
        "C. A previous direction worsened the fit:",
        "Avoid repeating the same direction and magnitude. Consider a smaller step, the opposite direction, or a different coherent perturbation.",
        "D. There is insufficient directional evidence:",
        "Propose a controlled exploratory perturbation over a broad, physically plausible layer group. Change few independent features so its effect can be interpreted after evaluation. In later iterations, use the history to test an opposite or alternative direction where useful.",
        "",
        "If several candidate directions are plausible, prefer the one best supported by local evaluated changes. When evidence is equally weak, favor an informative, bounded perturbation over an arbitrary complex modification."
        "",
        "(5). Physics-guided inference rules",
        # "1. Treat early gates as predominantly sensitive to shallower structure and later gates as increasingly sensitive to deeper structure, while remembering that AEM sensitivity is broad and there is no one-to-one gate-to-layer mapping.",
        "2. Favor coherent changes across neighboring layers over independent layer-by-layer edits.",
        "3. Prefer a smooth, minimum-structure profile unless the example and decay shape consistently support an interface or conductive/resistive body.",
        "4. Preserve supported conductive or resistive structures, smoothness does not require a monotonic profile."
        "5. Avoid unsupported one-layer spikes, rapid alternating high-low oscillations, and extreme deep-layer structure that is not constrained by the late-time response.",
        # "6. Do not copy an example merely because it is the closest available example. Equally, do not force the output to differ from an example solely to satisfy a non-copying rule. Any change should have an evidential basis."
        "",]
        # "--- EXAMPLES ---", ]

    # for i, ex in enumerate(demos):
    #     lines.append(f"Example {i} height: {ex['height_m']}")
    #     toks_d = ex["dBdt_code"]
    #     dstr = ",".join(str(x) for x in toks_d)
    #     lines.append(f"Example {i} dBdt_code: [{dstr}]")
    #     rhos = ex["rho_code"]
    #     rstr = ",".join(f"{x:.2f}" for x in rhos)
    #     lines.append(f"Example {i} rho_code: [{rstr}]")
    #     lines.append("")

    lines.extend([
        "--- QUERY CURVE ---",
    ])
    lines.append(f"Query height: {round(query.height, 3)}")
    dstr = ",".join(str(x) for x in response_codec.encode_row_norm(query.response))
    lines.append(f"Query dBdt_code: [{dstr}]")

    lines.extend([
        "",
        "--- EVALUATED HISTORY ---",
    ])
    for i, h in enumerate(history):
        lines.append(f"Iteration: {h.iteration}")
        rhos = h.candidate.model_code
        rstr = ",".join(f"{x:.2f}" for x in rhos)
        lines.append(f"Iteration {h.iteration} rho_code: [{rstr}]")
        toks_d = response_codec.encode_row_norm(h.candidate.predicted_response)
        dstr = ",".join(str(x) for x in toks_d)
        lines.append(f"Iteration {h.iteration} dBdt_code: [{dstr}]")

        lines.append(f"Iteration {h.iteration} normalized_residual: {h.candidate.misfit.normalized_residual.tolist()}")
        lines.append(f"Iteration {h.iteration} rms: {h.candidate.rms}")
        lines.append(f"Iteration {h.iteration} block_rms: {h.candidate.misfit.block_rms}")
        lines.append("")

    lines.extend([
        "",
        "--- THE BEST MODEL ---",
    ])
    best_rhos = best.model_code
    rstr = ",".join(f"{x:.2f}" for x in best_rhos)
    lines.append(f"Best rho_code: [{rstr}]")
    toks_d = response_codec.encode_row_norm(best.predicted_response)
    dstr = ",".join(str(x) for x in toks_d)
    lines.append(f"Best dBdt_code: [{dstr}]")

    lines.append(f"Best normalized_residual: {best.misfit.normalized_residual.tolist()}")
    lines.append(f"Best rms: {best.rms}")
    lines.append(f"Best block_rms: {best.misfit.block_rms}")
    lines.append("")

    lines.extend([
        "",
        "--- OUTPUT FORMAT ---",
        f"Return exactly {config.n_layer} log10(rho) values in [{np.log10(config.rho_min)}, {np.log10(config.rho_max)}].",
        "Provide your inferred rho_code as JSON in this exact format:",
        '{"rho_code": [rho1, rho2, rho3, ..., rhoN]}   // N equals number of layers',
        "",
        "Generate ONLY the JSON. No extra explanation text.",

    ])
    return "\n".join(lines)


def _group_info(config, constraint_model):
    n = operator.index(config.n_layer)
    if isinstance(config.n_layer, bool) or n <= 0:
        raise ValueError("n_layer must be a positive integer")
    ids = []
    for item in constraint_model.layer_group_ids:
        # Convert NumPy scalar labels without coercing different label types.
        gid = item.item() if hasattr(item, "item") else item
        if isinstance(gid, bool) or not isinstance(gid, (str, int, float)):
            raise ValueError("Group IDs must be scalar string or numeric labels")
        if isinstance(gid, float) and not math.isfinite(gid):
            raise ValueError("Group IDs must be finite")
        ids.append(gid)
    if len(ids) != n:
        raise ValueError(f"Expected {n} Layer group IDs, received {len(ids)}")
    groups = {}
    for layer, gid in enumerate(ids, start=1):
        groups.setdefault(gid, []).append(layer)
    rho_min, rho_max = float(config.rho_min), float(config.rho_max)
    if not (math.isfinite(rho_min) and math.isfinite(rho_max)
            and 0 < rho_min <= rho_max):
        raise ValueError("Require finite 0 < rho_min <= rho_max")
    return n, ids, groups, math.log10(rho_min), math.log10(rho_max)


def build_group_constraint_sections(config, constraint_model):
    """Return (intro_string, section_4_lines, output_format_lines)."""
    n, ids, groups, lower, upper = _group_info(config, constraint_model)
    symbols = {gid: f"theta_{j}" for j, gid in enumerate(groups)}
    group_table = [
        {"group_id": gid, "parameter": symbols[gid],
         "layers_1based": layers, "count": len(layers)}
        for gid, layers in groups.items()
    ]
    template = '{"rho_code": [' + ", ".join(symbols[g] for g in ids) + "]}"
    k = len(groups)
    intro = (
        f"IMPORTANT: This query has {n} model cells but only {k} independent "
        f"resistivity parameters, one per unique Layer group ID. Infer these {k} "
        f"parameters jointly, then copy them into the {n} output positions using "
        "the exact group mapping. Never estimate the output cells independently."
    )
    rules = [
        "(4). HARD structural constraints: shared group parameters",
        f"N = {n}; K = {k}. These values apply to this query.",
        "The grouping below is computed from the supplied Layer group IDs. "
        "Group IDs are categorical labels, not resistivity values or array indices.",
        "All layer numbers in this section are ONE-BASED output positions.",
        "Exact group-to-parameter mapping: " + json.dumps(
            group_table, ensure_ascii=False, separators=(",", ":")
        ),
        f"Infer exactly {k} group parameters. Each parameter is a finite "
        f"log10(rho) value in [{lower}, {upper}].",
        "The theta symbols are internal names assigned in order of first group "
        "occurrence. Their suffixes are not physical properties or group IDs.",
        "MANDATORY EXACT EQUALITIES:",
    ]
    for gid, layers in groups.items():
        equality = " == ".join(f"rho_code[{i}]" for i in layers)
        rules.append(f"{equality} == {symbols[gid]}")
    rules.extend([
        "For each group, infer ONE value and reuse it at ALL listed positions. "
        "Equality is exact, not approximate, including nonadjacent members.",
        "Different groups may have equal or different values. Do not require "
        "artificial contrast or a resistivity ordering between groups.",
        "Layer group IDs are authoritative for cell membership. A structural "
        "boundary occurs only where adjacent IDs differ, at their shared grid boundary.",
        "Constraint interface depths describe the original structure; mapped "
        "interface depths describe its completed mapping onto the resistivity grid. "
        "Do not remap depths or override the supplied IDs, including for rounding differences.",
        "Do not shift boundaries, change membership, split groups, introduce "
        "within-group gradients, or smooth/taper across boundaries.",
        "Example models are empirical evidence, not output-structure templates. "
        "Their cell-to-cell variations never override these equality constraints.",
        "Use all example responses and known flight heights to estimate the group "
        "parameters jointly. Height correction changes parameter estimates, not grouping.",
        "Optimize empirical agreement only within this group-parameter space. "
        "Do not generate an unconstrained cell-wise model and average it afterwards.",
        "Finalize and round each group parameter once, then copy that identical "
        "number to all its assigned positions. Do not modify cells after expansion.",
        "The following is a SYMBOLIC POSITION TEMPLATE, not numerical JSON output. "
        "Each repeated symbol must be replaced by exactly the same numerical value:",
        template,
        "A result violating any group equality is INVALID regardless of apparent response fit.",
        "",
    ])
    output = [
        "", "--- OUTPUT FORMAT: EXACT GROUP EXPANSION ---",
        f"Return one JSON object with only the key rho_code and exactly {n} numerical entries.",
        f"Every entry must be finite and in [{lower}, {upper}] in log10(rho) space.",
        "Use this symbolic position template; replace every symbol with its inferred number:",
        template,
        "Every occurrence of the same symbol MUST use the same numerical value "
        "and the same decimal representation. Preserve the displayed position order.",
        f"Before returning, internally check the length is {n}, all values are "
        "finite and in range, and every exact group equality in section (4) holds.",
        "If any check fails, reconstruct the array by copying the finalized group "
        "values into their assigned positions. Never repair cells independently.",
        "Return the fully expanded numerical JSON only. Do not output symbols, "
        "group parameters alone, expressions, ellipses, comments, Markdown, or explanations.",
    ]
    return intro, rules, output


def build_joint_prompt(query, examples, config, response_codec, constraint_model):
    tops, bottoms = layer_depths(config.layer_thicknesses, config.n_layer)
    layer_grid = [
        {
            "layer": i + 1,
            "top_m": round(float(tops[i]), 3),
            "bottom_m": None if not np.isfinite(bottoms[i]) else round(float(bottoms[i]), 3),
        }
        for i in range(config.n_layer)
    ]
    demos = []
    for rank, ex in enumerate(examples, start=1):
        demos.append(
            {
                "id": rank,
                "height_m": round(ex.height, 3),
                "dBdt_code": ex.response_code,
                "time": ex.time.tolist(),
                "rho_code": ex.model_code.tolist(),
            }
        )

    group_intro, group_rules, output_rules = build_group_constraint_sections(
        config,
        constraint_model,
    )

    lines = ["You are an expert geophysicist in airborne time-domain electromagnetic (AEM) joint inversion.",
             f"Your task is to infer a physically plausible {config.n_layer}-layer resistivity model for the query sounding using the supplied examples as local empirical evidence and the supplied structural constraint model as a mandatory model parameterization.",
             "You must perform the joint inversion using the provided structural constraint model to ensure that the resulting structure is consistent with the provided structural constraint model.",
             group_intro,
             "",
             "(1). Fixed acquisition configuration",
             "The transmitter waveform, loop geometry, receiver offset, response scaling, time gates, and layer thicknesses are fixed and identical for all examples and the query.",
             "Flight height may differ and is a known acquisition condition.",
             f"Each model contains {config.n_layer} values ordered from the shallowest layer to the deepest layer.",
             "rho_code definition:",
             "rho_code is a logarithmic resistivity representation:",
             "rho_code = log10(rho)",
             "dBdt_code definition:",
             "The dBdt_code is encoded from the response data using a log-10 min-max normalization process",
             r"dBdt\_code = round(clamp( \frac{\log_{10}(response) - log\_min}{log\_max - log\_min}, 0.0, 1.0) \times dBdt\_code\_max)",
             "The dBdt_code is a index-aligned representation, do not interpret them as time, depth, height, or resistivity.",
             "",
             "(2). Model parameterization:",
             f"The physical resistivity range is [{config.rho_min}, {config.rho_max}] Ohm-m."
             f"The corresponding model range is [{np.log10(config.rho_min)}, {np.log10(config.rho_max)}]."
             "Therefore:",
             f"rho_code = {np.log10(config.rho_min)} corresponds to {config.rho_min} Ohm-m",
             f"rho_code = {np.log10(config.rho_max)} corresponds to {config.rho_max} Ohm-m",
             "Every rho_code output value must be a finite number in rho_code = log10(rho) space, not physical Ohm-m.",
             "The Layer grid: " + _compact_json(layer_grid),
             "",
             "(3). How to use the examples",
             "Use every example as a local input-output relationship between:",
             "1. flight height",
             "2. the complete aligned response curve (dBdt_code),",
             f"3. and the corresponding {config.n_layer}-layer rho_code.",
             "Account for differences in flight height. Do not explain a response difference as a subsurface resistivity change when it can plausibly be associated with a height difference.",
             "Do not copy a example model unchanged unless both its response and flight height are effectively identical to the query.",
             "",
             *group_rules,
             "",
             "--- EXAMPLES ---", ]

    for i, ex in enumerate(demos):
        lines.append(f"Example {i} height: {ex['height_m']}")
        toks_d = ex["dBdt_code"]
        dstr = ",".join(str(x) for x in toks_d)
        lines.append(f"Example {i} dBdt_code: [{dstr}]")
        rhos = ex["rho_code"]
        rstr = ",".join(f"{x:.2f}" for x in rhos)
        lines.append(f"Example {i} rho_code: [{rstr}]")
        lines.append("")

    lines.extend([
        "--- CONSTRAINT MODEL ---",
    ])
    lstr = ",".join(str(x) for x in constraint_model.layer_group_ids)
    cstr = ",".join(str(x) for x in constraint_model.constraint_interface_depths)
    mstr = ",".join(str(x) for x in constraint_model.mapped_interface_depths)
    lines.append(f"Layer group IDs: [{lstr}]")
    lines.append(f"Constraint interface depths: [{cstr}]")
    lines.append(f"Mapped interface depths: [{mstr}]")

    lines.extend([
        "--- QUERY CURVE ---",
    ])
    lines.append(f"Query height: {round(query.height, 3)}")
    dstr = ",".join(str(x) for x in response_codec.encode_row_norm(query.response))
    lines.append(f"Query dBdt_code: [{dstr}]")

    lines.extend(output_rules)

    return "\n".join(lines)
