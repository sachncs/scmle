"""The decision layer.

Thresholds from a chartered cost model, net benefit with simultaneous
intervals, off-policy evaluation, fairness constraints, rollout design and an
anytime-valid promote, hold and rollback rule. No module here reads a sealed
set; the one-shot evaluations take the frozen rule and the final table as
arguments and refuse a second call.
"""
