"""Causal identification: select a strategy, estimate, falsify, tier.

The selector (`select`) is a pure function of the charter's causal block and
imports no estimator. Estimators are numpy and scipy only. Every gate that
uses a threshold has a measured operating characteristic in
`benchmarks/qualification/f10_certificates.json`; a gate whose certificate
missed its floor is descriptive.
"""
