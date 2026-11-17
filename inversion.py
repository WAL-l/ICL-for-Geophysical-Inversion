import numpy as np
from sympy import false
from ICL.config import InversionConfig
from ICL.encoding import ResponseCodec, ModelCodec
from ICL.inverter import AEMICLInverter
from ICL.data import AEMDataset, ConstraintModel
from dataclasses import dataclass
from ICL.data import Sounding
from retrieval.forward import forward
from retrieval.library import AEMSampleLibrary
from retrieval.retriever import HybridAEMRetriever
from matplotlib import pyplot as plt


def save_val_fig(rho, rho_pre, path):
    config = InversionConfig()

    rho = np.log10(rho)
    fig = plt.figure(figsize=(6, 9))
    gs = plt.GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
    ax = fig.add_subplot()
    x = np.asarray(rho).repeat(2)
    x_pre = np.asarray(rho_pre).repeat(2)
    edges = np.r_[0., np.cumsum(config.layer_thicknesses)]
    edges = np.r_[edges, edges[-1] + config.layer_thicknesses[-1]]
    depths = np.repeat(edges, 2)[1:-1]

    ax.invert_yaxis()
    ax.plot(x, depths, color="black", linewidth=1.5, label='Model')
    ax.plot(x_pre, depths, label='rho_pre')
    ax.set_title('(a)', fontdict={'fontsize': 10, 'fontweight': 'bold'})
    ax.set_ylabel("Depth (m)", fontsize=10, labelpad=1)
    ax.set_xlabel("Resistivity ( lg(ρ)(Ω·m) )", fontsize=10)
    ax.grid(axis='x', linestyle='--')
    ax.grid(axis='y', linestyle='--')
    plt.xticks(fontsize=10, )
    ax.legend()

    plt.show()


def plot_result(query, result, config, truth=None):

    initial = result.history[0].candidate
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    edges = np.r_[0., np.cumsum(config.layer_thicknesses)]
    edges = np.r_[edges, edges[-1] + config.layer_thicknesses[-1]]
    depths = np.repeat(edges, 2)[1:-1]
    for label, rho in (("Initial", initial.resistivity),
                       ("Best", result.best.resistivity), ("Reference", truth)):
        if rho is not None:
            axes[0].plot(np.repeat(np.log10(rho), 2), depths, label=label)
    if result.constraint_model is not None:
        for i, depth in enumerate(result.constraint_model.mapped_interface_depths):
            axes[0].axhline(depth, color="tab:purple", linestyle=":", alpha=0.6,
                            label="Velocity interface on AEM grid" if i == 0 else None)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="log10 resistivity (Ohm-m)", ylabel="Depth (m)")
    axes[1].loglog(query.times, query.response, "k.-", label="Observed")
    axes[1].loglog(query.times, initial.predicted_response, "--", label="Initial")
    axes[1].loglog(query.times, result.best.predicted_response, label="Best")
    axes[1].set(xlabel="Time (s)", ylabel="Response (forward solver units)")
    evaluated = [record for record in result.history if record.candidate is not None]
    axes[2].plot([record.iteration for record in evaluated],
                 [record.candidate.rms for record in evaluated], "o-", label="Candidate RMS")
    axes[2].plot([record.iteration for record in evaluated],
                 np.minimum.accumulate([record.candidate.rms for record in evaluated]),
                 "s--", label="Best RMS")
    axes[2].axhline(config.target_rms, color="gray", linestyle=":", label="Target")
    axes[2].set(xlabel="Iteration (0 = initial)", ylabel="Normalized RMS")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle(f"{query.sounding_id}: {result.stop_reason}")
    fig.tight_layout()
    return fig

@dataclass(frozen=True, slots=True)
class ExampleCode:
    response_code: np.ndarray
    model_code: np.ndarray
    height: float
    time: np.ndarray

def run_loop():
    config = InversionConfig()
    data = np.load("data/filed/data.npy")
    field = np.load("data/filed/filed_data.npy")
    heights = np.load("data/filed/h.npy")
    rhos = np.load("data/filed/rho.npy")
    response_codec = ResponseCodec.fit(data)
    model_codec = ModelCodec(config.rho_min, config.rho_max)


    index = int(np.flatnonzero((heights > 30) & (heights < 44))[50])
    raw_response = field[index]
    valid_mask = raw_response > 0
    query_times = config.times[valid_mask]
    query = Sounding(
        response=raw_response[valid_mask].copy(), height=float(heights[index]),
        times=query_times.copy(), sounding_id=f"sample_{index}",
    )
    library = AEMSampleLibrary.load("tests/retrieval_data", mmap_mode=None)
    retrieved = HybridAEMRetriever(library).search(
        np.log10(query.response), query.height, top_k=5)
    examples = [
        ExampleCode(
            response_code=response_codec.encode_row_norm(item.response[valid_mask]),
            model_code=model_codec.encode_log(item.model),
            height=item.height_m, time=query_times.copy(),
        )
        for item in retrieved
    ]
    inverter = AEMICLInverter(forward, config)
    result = inverter.loop_invert(query, examples, response_codec)
    for record in result.history:
        if record.candidate is not None:
            print(f"iteration={record.iteration}, RMS={record.candidate.rms:.6f}, "
                  f"improved={record.improved}")
        if record.error:
            print(f"iteration={record.iteration}, error={record.error}")
    print(f"Best iteration: ; stop reason: {result.stop_reason}")

    fig = plot_result(query, result, config, truth=rhos[index])
    plt.show()
    plt.close(fig)
    return result

def run_joint():
    config = InversionConfig()
    data = np.load("./data/data.npy")
    heights = np.load("./data/h.npy")
    rhos = np.load("./data/rho.npy")
    dataset = AEMDataset(data,heights,rhos,config.times)
    response_codec = ResponseCodec.fit(dataset.responses)
    model_codec = ModelCodec(config.rho_min, config.rho_max)

    inverter = AEMICLInverter(
        forward_operator=forward,
        config=config,
    )

    library = AEMSampleLibrary.load("./tests/retrieval_data", mmap_mode=None)

    idx = 117
    response_raw = data[idx]
    valid_mask = response_raw > 0
    f_response = response_raw[valid_mask]

    response = np.log10(np.maximum(f_response, 1e-30))
    retriever = HybridAEMRetriever(library)
    retrieves = retriever.search(response, heights[idx],3)
    examples = []
    for retrieve in retrieves:

        tok_d = response_codec.encode_row_norm(retrieve.response[valid_mask])
        tok_m = model_codec.encode_log(retrieve.model)

        examples.append(
            ExampleCode(
                response_code=tok_d,
                model_code=tok_m,
                height=retrieve.height_m,
                time=config.times,
            )
        )

    sounding = dataset.sounding(idx,false)
    constraint_interfaces = np.array([80, 170], dtype=float)
    constraint_model = ConstraintModel.from_constraint_model(config.layer_thicknesses,constraint_interfaces)

    result = inverter.joint_invert(sounding, examples,response_codec, constraint_model)
    save_val_fig(rhos[idx], result, f"outputs/")


if __name__ == "__main__":
    run_joint()