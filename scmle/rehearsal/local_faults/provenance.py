"""Scenarios that break provenance: signatures, events, lineage, approvals."""

import dataclasses

from scmle.contracts import lineage
from scmle.contracts import reliability
from scmle.data.readers import incremental
import scmle.data.readers.registry
from scmle.governance import attestation
import scmle.governance.registry
import scmle.governance.reliability
from scmle.rehearsal.local_faults import workspace


class Provenance(workspace.Workspace):
    """Bad signatures, duplicate events, lineage gaps, rejections, snapshots."""

    __slots__ = ()

    def run_invalid_signature(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Tampers with an attestation signature and records the failed check.

        Builds a signed release attestation, replaces its signature with
        zeros, verifies it with the same key, and writes a quarantine record
        ``quarantine-signature.json`` under the root.

        Returns:
            The three evidence artifacts, in order: signature tampering,
            verification failure and quarantine record.
        """
        scenario = reliability.Contingency.INVALID_SIGNATURE
        provenance = {
            "run_id": "local-signature",
            "model": {"manifest_sha256": "manifest"},
            "signature": {"present": True},
        }
        record = attestation.Attestation.from_records(
            provenance, {}, signing_key="local-key"
        )
        tampered = dataclasses.replace(record, signature="0" * 64)
        rejected = not tampered.verify(provenance, {}, signing_key="local-key")
        quarantine = self.root / "quarantine-signature.json"
        self.write_json(
            quarantine, {"state": "quarantined", "rejected": rejected}
        )
        return (
            self.artifact(
                scenario,
                "signature_tampering",
                "attestation signature was changed before verification",
                {"rejected": rejected},
            ),
            self.artifact(
                scenario,
                "verification_failure",
                "tampered attestation failed cryptographic verification",
                {"verified": not rejected},
            ),
            self.artifact(
                scenario,
                "quarantine_record",
                "untrusted attestation was quarantined",
                {"quarantined": quarantine.exists()},
            ),
        )

    def run_duplicate_event(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Commits one event twice to an ingestion ledger and records both.

        Uses an incremental ingestion ledger stored in
        ``duplicate-events.json`` under the root. The evidence reports the
        accepted and duplicate identities of the two commits, the events the
        writer received, and the identities the ledger holds afterwards.

        Returns:
            The three evidence artifacts, in order: duplicate event,
            idempotency key and state transition.
        """
        scenario = reliability.Contingency.DUPLICATE_EVENT
        path = self.root / "duplicate-events.json"
        ledger = incremental.Ledger(path, "local-source")
        event = incremental.Event(
            "event-1", 1, "2026-10-20T10:00:00+00:00", {"value": 1}
        )
        accepted: list[str] = []

        def accept(
            events: tuple[incremental.Event, ...],
        ) -> None:
            accepted.extend(item.identifier for item in events)

        first = ledger.finalize((event,), accept)
        second = ledger.finalize((event,), accept)
        return (
            self.artifact(
                scenario,
                "duplicate_event",
                "the same event identity was submitted twice",
                {
                    "first": first.accepted,
                    "second": second.duplicates,
                },
            ),
            self.artifact(
                scenario,
                "idempotency_key",
                "event identity was used as the idempotency key",
                {"accepted": accepted},
            ),
            self.artifact(
                scenario,
                "state_transition",
                "duplicate submission left one committed event",
                {"committed": ledger.load().seen},
            ),
        )

    def run_incomplete_lineage(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Adds a lineage edge to a missing node and records the rejection.

        The graph holds one raw-source node; an edge to an absent node is
        added and `ValueError` is expected. The release-block and audit
        artifacts are fixed statements.

        Returns:
            The three evidence artifacts, in order: lineage gap, release
            block and audit record.
        """
        scenario = reliability.Contingency.INCOMPLETE_LINEAGE
        graph = lineage.Graph(
            (lineage.Node("raw", lineage.Kind.RAW_SOURCE, "raw"),)
        )
        rejected = False
        try:
            graph.connect(lineage.Edge("raw", "curation", "produced"))
        except ValueError:
            # handled: expected lineage rejection recorded in lineage_gap
            # evidence
            rejected = True
        return (
            self.artifact(
                scenario,
                "lineage_gap",
                "missing lifecycle node was detected before graph publication",
                {"rejected": rejected},
            ),
            self.artifact(
                scenario,
                "release_block",
                "incomplete lineage blocked release",
                {"release_ready": False},
            ),
            self.artifact(
                scenario,
                "audit_record",
                "lineage rejection was retained as an audit observation",
                {"node": "curation"},
            ),
        )

    def run_rejected_approval(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Validates a signed rejected approval and records the refusal.

        Builds and signs a reviewer approval whose decision is rejected and
        checks it against its own target; `ValueError` is expected. No
        registry or alias is involved.

        Returns:
            The three evidence artifacts, in order: approval rejection,
            release block and audit record.
        """
        scenario = reliability.Contingency.REJECTED_APPROVAL
        approval = scmle.governance.registry.Sanction(
            model="local-model",
            version=1,
            content_id="bundle-digest",
            reviewer="independent-reviewer",
            decision=scmle.governance.registry.Verdict.REJECTED,
            reason="controlled rejection",
            evidence=("artifact://review",),
            approved_at="2026-10-20T10:00:00+00:00",
        ).sign("local-key")
        rejected = False
        try:
            approval.validate(
                "local-model", 1, "bundle-digest", signing_key="local-key"
            )
        except ValueError:
            # handled: expected approval refusal recorded in approval_rejection
            # evidence
            rejected = True
        return (
            self.artifact(
                scenario,
                "approval_rejection",
                "reviewer rejection was evaluated before promotion",
                {"rejected": rejected},
            ),
            self.artifact(
                scenario,
                "release_block",
                "rejected approval prevented alias movement",
                {"release_ready": False},
            ),
            self.artifact(
                scenario,
                "audit_record",
                "rejected approval remained reviewable",
                {"reviewer": approval.reviewer},
            ),
        )

    def run_changed_snapshot(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Registers a snapshot, then one with the same id and other data.

        The second registration in an in-memory source registry is expected
        to raise `ValueError`; the outcome is recorded. The lineage artifact
        is a fixed statement.

        Returns:
            The three evidence artifacts, in order: snapshot digest change,
            run invalidation and lineage record.
        """
        scenario = reliability.Contingency.CHANGED_SNAPSHOT
        previous = scmle.data.readers.registry.Observation(
            source="source",
            uri="file:///source.csv",
            snapshot="snapshot-1",
            digests={"schema": "schema-1", "data": "data-1", "raw": "raw-1"},
            rows=1,
            layout={"value": "int"},
            captured="2026-10-20T10:00:00+00:00",
        )
        registry = scmle.data.readers.registry.Sources()
        registry.register(previous)
        changed = dataclasses.replace(
            previous, digests={**previous.digests, "data": "data-2"}
        )
        rejected = False
        try:
            registry.register(changed)
        except ValueError:
            # handled: expected snapshot conflict recorded in run_invalidation
            # evidence
            rejected = True
        return (
            self.artifact(
                scenario,
                "snapshot_digest_change",
                "the same snapshot ID was presented with changed data",
                {"snapshot_id": changed.snapshot},
            ),
            self.artifact(
                scenario,
                "run_invalidation",
                "conflicting snapshot evidence was rejected",
                {"rejected": rejected},
            ),
            self.artifact(
                scenario,
                "lineage_record",
                "changed snapshot evidence blocked reuse",
                {"release_ready": False},
            ),
        )
