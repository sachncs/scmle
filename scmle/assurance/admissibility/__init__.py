"""Admissibility and provenance of fields.

Decides which fields may enter a model on facts about how each value comes to
exist, and uses statistics only to raise suspicion about what provenance
cannot settle. Every statistical test returns a `Check` for the hypothesis
ledger; no test here proves a field leaks or that a graph is right.
"""
