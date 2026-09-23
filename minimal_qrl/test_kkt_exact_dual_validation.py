import numpy as np

from minimal_qrl.industry_exp.kkt_exact_dual_validation import (
    exact_dual_reaction_audit,
)


def test_exact_per_edge_dual_cancels_global_push_on_chain():
    result = exact_dual_reaction_audit(
        values=np.asarray([2.0, 1.0, 0.0]),
        sources=np.asarray([0, 1, 2]),
        destinations=np.asarray([1, 2, -1]),
        costs=np.asarray([1.0, 1.0, 0.0]),
        edge_duals=np.asarray([0.5, 1.0, 0.0]),
        u_trap_mask=np.asarray([True, True, False]),
    )

    assert result["primal"]["max_constraint_violation"] == 0.0
    assert result["complementarity"]["linf"] == 0.0
    assert result["stationarity"]["residual_l2"] < 1e-12
    assert result["stationarity"]["u_region_residual_l2"] < 1e-12
    density = result["dual_scaling"]["uniform_expectation_density_positive"]
    assert np.isclose(density["max"], 3.0)
