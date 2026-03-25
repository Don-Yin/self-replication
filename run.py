"""single entry point for the self-replication phase diagram experiment."""
import importlib
import json
import logging
import time
from pathlib import Path

from src.config import RESULTS, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def run_stage(label: str, module_path: str, artifact: Path, **kwargs) -> dict | None:
    """import and run a pipeline stage only if its artifact is missing."""
    if artifact.exists():
        logger.info("skipping %s (exists: %s)", label, artifact.name)
        return json.loads(artifact.read_text()) if artifact.suffix == ".json" else None
    logger.info("=== %s ===", label)
    t = time.time()
    module = importlib.import_module(module_path)
    result = module.run(artifact=artifact, **kwargs)
    logger.info("  done in %.1f sec", time.time() - t)
    return result


def main():
    """run full pipeline with lazy execution."""
    ensure_dirs()
    output = {}

    # phase A: pipeline validation
    output["validation"] = run_stage(
        "pipeline validation",
        "jobs.validate_pipeline",
        artifact=RESULTS / "validation.json",
    )

    # phase B: exhaustive censuses (k=2)
    output["k2_vn"] = run_stage(
        "k=2 von Neumann census (1024 rules)",
        "jobs.k2_census",
        artifact=RESULTS / "k2-vn-census.json",
        neighborhood="von_neumann",
    )

    output["k2_moore"] = run_stage(
        "k=2 Moore census (262K rules)",
        "jobs.k2_moore_census",
        artifact=RESULTS / "k2-moore-census.json",
    )

    # phase C: boundary analysis
    output["boundary_measures"] = run_stage(
        "boundary measures (mass-balance, entropy)",
        "jobs.boundary_measures",
        artifact=RESULTS / "k2-moore-boundary-measures.json",
        census_path=RESULTS / "k2-moore-census.json",
    )

    output["oinfo_boundary"] = run_stage(
        "O-information at boundary",
        "jobs.oinfo_boundary",
        artifact=RESULTS / "k2-moore-oinfo-boundary.json",
        census_path=RESULTS / "k2-moore-census.json",
    )

    output["tier2_check"] = run_stage(
        "tier-2 check on tier-1 rules",
        "jobs.tier2_check",
        artifact=RESULTS / "k2-moore-tier2-check.json",
        census_path=RESULTS / "k2-moore-census.json",
    )

    # phase D: extensions
    output["k3_sweep"] = run_stage(
        "k=3 Moore sweep (10K rules)",
        "jobs.k3_sweep",
        artifact=RESULTS / "k3-moore-sweep.json",
        neighborhood="moore",
    )

    output["c4_sample"] = run_stage(
        "k=2 C4 rotationally-symmetric sample (10K rules)",
        "jobs.k2_c4_sample",
        artifact=RESULTS / "k2-c4-sample.json",
    )

    output["tier3_causal"] = run_stage(
        "tier-3 causal perturbation test (1000 rules)",
        "jobs.tier3_causal",
        artifact=RESULTS / "k2-moore-tier3-causal.json",
        tier2_path=RESULTS / "k2-moore-tier2-check.json",
        n_sample=1000,
    )

    output["ext_moore"] = run_stage(
        "k=2 extended Moore (|N|=25) sweep (10K rules)",
        "jobs.extended_moore_sweep",
        artifact=RESULTS / "k2-extmoore-sweep.json",
    )

    output["k2_vn_eq"] = run_stage(
        "k=2 vN equalised protocol (1024 rules)",
        "jobs.k2_vn_equalized",
        artifact=RESULTS / "k2-vn-equalized.json",
    )

    # phase D2: new analyses
    output["derrida_phase"] = run_stage(
        "Derrida phase diagram (~1000 rules)",
        "jobs.derrida_phase_diagram",
        artifact=RESULTS / "k2-moore-derrida-phase.json",
        census_path=RESULTS / "k2-moore-census.json",
    )

    output["f_ablation"] = run_stage(
        "F-weighting ablation (uniform/linear/quadratic)",
        "jobs.f_ablation",
        artifact=RESULTS / "k2-moore-f-ablation.json",
        census_path=RESULTS / "k2-moore-census.json",
    )

    output["cross_mass_balance"] = run_stage(
        "cross-substrate mass-balance (vN + k=3)",
        "jobs.mass_balance_cross_substrate",
        artifact=RESULTS / "cross-substrate-mass-balance.json",
        vn_census_path=RESULTS / "k2-vn-census.json",
        k3_sweep_path=RESULTS / "k3-moore-sweep.json",
    )

    output["asal_comparison"] = run_stage(
        "ASAL open-endedness cross-comparison",
        "jobs.asal_comparison",
        artifact=RESULTS / "k2-moore-asal-comparison.json",
        census_path=RESULTS / "k2-moore-census.json",
    )

    output["sensitivity"] = run_stage(
        "detection-threshold sensitivity (5K rules x 12 combos)",
        "jobs.sensitivity_sweep",
        artifact=RESULTS / "k2-moore-sensitivity.json",
        census_path=RESULTS / "k2-moore-census.json",
    )

    # phase E: statistical controls
    output["matched_controls"] = run_stage(
        "lambda/F-matched controls + permutation test",
        "jobs.matched_controls",
        artifact=RESULTS / "k2-moore-matched-controls.json",
        boundary_path=RESULTS / "k2-moore-boundary-measures.json",
        oinfo_path=RESULTS / "k2-moore-oinfo-boundary.json",
    )

    output["effect_sizes"] = run_stage(
        "effect sizes (Cohen's d, rank-biserial)",
        "jobs.effect_sizes",
        artifact=RESULTS / "k2-moore-effect-sizes.json",
        boundary_path=RESULTS / "k2-moore-boundary-measures.json",
        oinfo_path=RESULTS / "k2-moore-oinfo-boundary.json",
    )

    output["f_marginal"] = run_stage(
        "F-marginal (replication rate vs F)",
        "jobs.f_marginal",
        artifact=RESULTS / "k2-moore-f-marginal.json",
        census_path=RESULTS / "k2-moore-census.json",
    )

    output["logistic_classifier"] = run_stage(
        "logistic classifier + ROC",
        "jobs.logistic_classifier",
        artifact=RESULTS / "k2-moore-logistic.json",
        boundary_path=RESULTS / "k2-moore-boundary-measures.json",
        oinfo_path=RESULTS / "k2-moore-oinfo-boundary.json",
    )

    # phase F: verification
    output["verification"] = run_stage(
        "256x256 verification (100 rules)",
        "jobs.verification_run",
        artifact=RESULTS / "k2-moore-verification.json",
        census_path=RESULTS / "k2-moore-census.json",
    )

    # phase F: visualizations
    output["replicator_viz"] = run_stage(
        "replicator visualization",
        "jobs.replicator_viz",
        artifact=RESULTS / "replicator-viz.json",
        census_path=RESULTS / "k2-moore-census.json",
    )

    run_stage(
        "all publication figures",
        "jobs.figures",
        artifact=RESULTS / ".figures-done",
    )

    run_stage(
        "new analysis figures (Derrida, F-ablation, cross-substrate)",
        "jobs.new_figures",
        artifact=RESULTS / ".new-figures-done",
    )

    run_stage(
        "tier detection examples figure",
        "jobs.tier_examples_viz",
        artifact=RESULTS / "tier-examples-viz.json",
    )

    # save combined results
    combined = RESULTS / "all-experiments.json"
    combined.write_text(json.dumps(output, indent=2, default=str))
    logger.info("all results saved to %s", combined)


if __name__ == "__main__":
    main()
