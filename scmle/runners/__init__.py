"""Portable training: one job spec, a `Runner` per place it can run.

`base` defines the job spec, the run manifest and the `Runner` template
(submit, supervise, retry, enforce a cost budget, write the manifest to
durable storage); `local` is the adapter that ships (others implement
`Runner`); `roles` separates what training and serving may touch;
`checkpoint` holds the resumable state; `project` runs an `MLProject` as the
workload.
"""
