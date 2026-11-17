"""The claim "deployed" and the reconciliation behind it."""

from typing import Any

from scmle.deploy import observation
from scmle.deploy.release_state import promotion
from scmle.deploy.release_state import rollback
from scmle.deploy.release_state import status
from scmle.governance import deployment_ledger
from scmle.governance import registry


class ReleaseRegistry(promotion.Promotion, rollback.Rollback):
    """Release state, promotion, rollback and reconciliation for a registry."""

    def live(
        self,
        model: str,
        environment: str,
        seen: observation.Observation,
    ) -> int:
        """Returns the version the registry says is deployed, if confirmed.

        Args:
            model: Model name.
            environment: Environment name.
            seen: What the platform reports.

        Returns:
            The version recorded for the environment.

        Raises:
            RuntimeDisagrees: If the registry records no deployment or any
                finding of `reconcile` applies.
        """
        report = self.reconcile(model, environment, seen)
        recorded = self.environment(model, environment)
        if recorded is None or not report.consistent:
            raise status.RuntimeDisagrees(
                f"{model} is not confirmed deployed in {environment}",
                {"findings": [f.to_dict() for f in report.findings]},
            )
        return int(recorded["version"])

    # -- reconcile ---------------------------------------------------------

    def recall(self, model: str, environment: str) -> dict[str, Any] | None:
        """Returns the newest confirmed ledger event for the environment.

        Args:
            model: Model name.
            environment: Environment name.

        Returns:
            A copy of the newest promoted or rolled-back event for the model
            and environment, or None if there is no ledger or no such event.
        """
        if self.ledger is None:
            return None
        for row in reversed(self.ledger.rows):
            body = row["body"]
            if (
                row["kind"] == deployment_ledger.DEPLOYMENT
                and body.get("model") == model
                and body.get("environment") == environment
                and body.get("event") in ("promoted", "rolled_back")
            ):
                return dict(body)
        return None

    def compare(
        self,
        model: str,
        environment: str,
        recorded: dict[str, Any],
        seen: observation.Observation,
    ) -> list[status.Finding]:
        """Compares what the registry records with a platform observation.

        Args:
            model: Model name.
            environment: Environment name.
            recorded: What the registry records as running there.
            seen: What the platform reports.

        Returns:
            One finding per disagreement in revision, digest, traffic share or
            release state.
        """
        findings: list[status.Finding] = []
        if seen.revision != recorded["revision"]:
            findings.append(
                status.Finding(
                    "revision_mismatch",
                    f"registry says {recorded['revision']}, the platform "
                    f"serves {seen.revision}",
                )
            )
        if seen.image_digest != recorded["image_digest"]:
            findings.append(
                status.Finding(
                    "digest_mismatch",
                    f"registry says {recorded['image_digest']}, the "
                    f"platform runs {seen.image_digest}",
                )
            )
        share = seen.portion(recorded["revision"])
        if share != status.FULL_ROLLOUT:
            findings.append(
                status.Finding(
                    "traffic_split",
                    f"the recorded revision holds {share}% of traffic",
                )
            )
        known = self.states(model)
        production = environment == status.PRODUCTION_ENVIRONMENT
        if (
            production
            and known.get(recorded["version"])
            is not status.ReleaseState.PRODUCTION
        ):
            findings.append(
                status.Finding(
                    "state_mismatch",
                    f"v{recorded['version']} is deployed but its state "
                    f"is {known.get(recorded['version'], 'unknown')}",
                )
            )
        for number, state in known.items():
            if (
                production
                and state is status.ReleaseState.PRODUCTION
                and number != recorded["version"]
            ):
                findings.append(
                    status.Finding(
                        "state_mismatch",
                        f"v{number} is in production state but the "
                        f"environment records v{recorded['version']}",
                    )
                )
        return findings

    def reconcile(
        self,
        model: str,
        environment: str,
        seen: observation.Observation,
    ) -> status.Reconciliation:
        """Compares the registry and the ledger with a platform observation.

        Every disagreement is a finding; none is repaired. The finding kinds
        are `no_registry_record`, `revision_mismatch`, `digest_mismatch`,
        `traffic_split`, `state_mismatch`, `not_ready`, `alias_not_deployed`,
        `ledger_missing` and `ledger_disagrees`.

        Args:
            model: Model name.
            environment: Environment name.
            seen: What the platform reports.

        Returns:
            The report listing every disagreement; consistent if there is none.
        """
        findings: list[status.Finding] = []
        recorded = self.environment(model, environment)
        if recorded is None:
            findings.append(
                status.Finding(
                    "no_registry_record",
                    f"the platform runs {seen.image_digest} but the registry "
                    f"records no deployment of {model} in {environment}",
                )
            )
        else:
            findings.extend(self.compare(model, environment, recorded, seen))
        if not seen.ready:
            findings.append(
                status.Finding(
                    "not_ready", "the platform does not report ready"
                )
            )
        if environment == status.PRODUCTION_ENVIRONMENT:
            champion = self.store.aliases(model).get(registry.CHAMPION)
            expected = None if recorded is None else recorded["version"]
            if champion is not None and champion != expected:
                findings.append(
                    status.Finding(
                        "alias_not_deployed",
                        f"the champion alias points at v{champion}, which the "
                        f"runtime has not confirmed (confirmed: {expected})",
                    )
                )
        event = self.recall(model, environment)
        if self.ledger is not None:
            if event is None and recorded is not None:
                findings.append(
                    status.Finding(
                        "ledger_missing",
                        "the ledger has no confirmed deployment event",
                    )
                )
            elif event is not None and (
                recorded is None
                or (event["version"], event["revision"], event["image_digest"])
                != (
                    recorded["version"],
                    recorded["revision"],
                    recorded["image_digest"],
                )
            ):
                findings.append(
                    status.Finding(
                        "ledger_disagrees",
                        f"the ledger's last event is v{event['version']} "
                        f"{event['revision']}",
                    )
                )
        return status.Reconciliation(model, environment, tuple(findings))
