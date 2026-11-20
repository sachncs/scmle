"""Commands that govern a model's life: loop, claims and trail."""

import argparse
import json
import logging
import os
import pathlib
import types

import pandas

from scmle.artifacts import claims_audit
from scmle.cli import command
from scmle.contracts import errors
from scmle.contracts import signing
from scmle.governance import deployment_ledger
import scmle.governance.loop
from scmle.trail import builder
from scmle.trail import chain
import scmle.trail.show

LOGGER = logging.getLogger(__name__)


class Loop(command.Command):
    """Drives the closed loop of a deployment (observe, step, ask, reply)."""

    name = "loop"
    help = "observe batches, step the closed loop, list and answer escalations"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the loop actions.

        Args:
            parser: The argument parser of this command.
        """
        actions = parser.add_subparsers(dest="loop_action", required=True)
        init = actions.add_parser("init", help="create the deployment ledger")
        init.add_argument("deploy_dir")
        init.add_argument("--policy", required=True, help="policy JSON file")
        init.add_argument("--charter-hash", required=True)
        observe = actions.add_parser("observe", help="route a batch of cases")
        observe.add_argument("deploy_dir")
        observe.add_argument("--batch", required=True, help="CSV of cases")
        observe.add_argument("--predictions-column", default="prediction")
        observe.add_argument("--decisions-column", default="decision")
        mature = actions.add_parser("mature", help="record arrived losses")
        mature.add_argument("deploy_dir")
        mature.add_argument(
            "--labels", required=True, help="CSV with case_id and loss"
        )
        step = actions.add_parser("step", help="apply the decision map")
        step.add_argument("deploy_dir")
        pending = actions.add_parser("pending", help="list open escalations")
        pending.add_argument("deploy_dir")
        reply = actions.add_parser("reply", help="record a signed reply")
        reply.add_argument("deploy_dir")
        reply.add_argument("escalation_id")
        reply.add_argument(
            "--attestation", required=True, help="attestation JSON file"
        )
        reply.add_argument(
            "--key-file",
            required=True,
            help="file with the key that verifies the reply (the person's)",
        )
        verify = actions.add_parser("verify", help="verify the ledger chain")
        verify.add_argument("deploy_dir")

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status of the chosen action.
        """
        action = getattr(self, f"run_{args.loop_action}")
        status: int = action(args)
        return status

    @staticmethod
    def run_init(args: argparse.Namespace) -> int:
        """Opens the deployment, creating its ledger on first use.

        The ledger is created from the policy file and the charter hash.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        policy = scmle.governance.loop.Preregistration.from_dict(
            json.loads(pathlib.Path(args.policy).read_text(encoding="utf-8"))
        )
        charter = types.SimpleNamespace(charter_hash=args.charter_hash)
        scmle.governance.loop.Loop.open(args.deploy_dir, charter, policy)
        LOGGER.info("opened deployment ledger in %s", args.deploy_dir)
        return command.EXIT_OK

    @staticmethod
    def run_observe(args: argparse.Namespace) -> int:
        """Routes one batch and prints how many cases went where.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        loop = scmle.governance.loop.Loop.resume(args.deploy_dir)
        frame = pandas.read_csv(args.batch)
        outcome = loop.observe(
            frame,
            frame[args.predictions_column].tolist(),
            frame[args.decisions_column].tolist(),
        )
        counts = pandas.Series(outcome.routing.status).value_counts().to_dict()
        print(json.dumps({"batch": outcome.batch, "routing": counts}))
        return command.EXIT_OK

    @staticmethod
    def run_mature(args: argparse.Namespace) -> int:
        """Stores arrived losses by case id.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        loop = scmle.governance.loop.Loop.resume(args.deploy_dir)
        frame = pandas.read_csv(args.labels)
        matched = loop.mature(
            dict(zip(frame["case_id"].astype(str), frame["loss"], strict=True))
        )
        print(json.dumps({"matched": matched}))
        return command.EXIT_OK

    @staticmethod
    def run_step(args: argparse.Namespace) -> int:
        """Steps the loop and prints what it found.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        loop = scmle.governance.loop.Loop.resume(args.deploy_dir)
        result = loop.step()
        print(
            json.dumps(
                {
                    "status": result.status,
                    "mechanism": result.mechanism.value,
                    "events": [list(event) for event in result.events],
                    "escalation": (
                        None
                        if result.escalation is None
                        else result.escalation.id
                    ),
                }
            )
        )
        return command.EXIT_OK

    @staticmethod
    def run_pending(args: argparse.Namespace) -> int:
        """Lists open escalations with their questions.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        loop = scmle.governance.loop.Loop.resume(args.deploy_dir)
        print(
            json.dumps(
                [
                    {
                        "id": e.id,
                        "question": e.question,
                        "question_hash": e.question_hash,
                        "evidence": list(e.evidence),
                        "proposed_default": e.proposed_default,
                    }
                    for e in loop.survey()
                ],
                indent=2,
            )
        )
        return command.EXIT_OK

    @staticmethod
    def run_reply(args: argparse.Namespace) -> int:
        """Records a person's signed reply; refused replies fail.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).

        Raises:
            ConfigError: If the reply is refused.
        """
        loop = scmle.governance.loop.Loop.resume(args.deploy_dir)
        reply = deployment_ledger.Attestation.from_dict(
            json.loads(
                pathlib.Path(args.attestation).read_text(encoding="utf-8")
            )
        )
        key = pathlib.Path(args.key_file).read_text(encoding="utf-8").strip()
        try:
            outcome = loop.reply(args.escalation_id, reply, key)
        except scmle.governance.loop.RefusalError as error:
            raise errors.ConfigError(f"reply refused: {error}") from error
        print(
            json.dumps(
                {
                    "tier": outcome.tier.value,
                    "decision": outcome.decision.value,
                    "safe_default_applies": outcome.safe_default_applies,
                }
            )
        )
        return command.EXIT_OK

    @staticmethod
    def run_verify(args: argparse.Namespace) -> int:
        """Verifies the ledger chain and the e-LOND replay.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).

        Raises:
            ArtifactError: If the ledger fails verification.
        """
        path = pathlib.Path(args.deploy_dir) / deployment_ledger.FILE_NAME
        try:
            ledger = deployment_ledger.Ledger.load(path)
        except deployment_ledger.MisuseError as error:
            raise errors.ArtifactError(
                f"ledger verification failed: {error}"
            ) from error
        print(
            json.dumps(
                {"ok": True, "rows": len(ledger.rows), "head": ledger.head}
            )
        )
        return command.EXIT_OK


class Claims(command.Command):
    """Lists, shows and verifies the claims of a charter run."""

    name = "claims"
    help = "list, show or verify the claims and ledger of a charter run"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the `list`, `show` and `verify` actions.

        Args:
            parser: The argument parser of this command.
        """
        actions = parser.add_subparsers(dest="claims_action", required=True)
        listing = actions.add_parser("list", help="one row per claim")
        listing.add_argument("run_dir")
        listing.add_argument("--tier", choices=["confirmed", "unverified"])
        listing.add_argument("--json", action="store_true")
        show = actions.add_parser("show", help="everything about one claim")
        show.add_argument("run_dir")
        show.add_argument("claim_id")
        verify = actions.add_parser(
            "verify",
            help="check the ledger chain and replay, the claims file, the "
            "second-stage accounting, the custody log and the charter",
        )
        verify.add_argument("run_dir")
        verify.add_argument("--json", action="store_true")
        verify.add_argument(
            "--key-file",
            help="file holding the signing key; default: SCMLE_SIGNING_KEY",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status of the chosen action.
        """
        action = getattr(self, f"run_{args.claims_action}")
        status: int = action(args)
        return status

    @staticmethod
    def run_list(args: argparse.Namespace) -> int:
        """Prints the claims of a run, optionally only those of one tier.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        rows = claims_audit.inventory(args.run_dir)
        if args.tier:
            rows = [r for r in rows if r["tier"] == args.tier]
        if args.json:
            print(json.dumps(rows, indent=2))
            return command.EXIT_OK
        for row in rows:
            if row.get("kind"):
                print(
                    f"{row['claim_id']}  {row['tier']:<10}  "
                    f"registered {row['kind']}  status={row['status']}"
                    + (
                        f"  identification={row['identification']}"
                        if row["identification"]
                        else ""
                    )
                )
                continue
            adjusted = row["adjusted_p_value"]
            print(
                f"{row['claim_id']}  {row['tier']:<10}  "
                f"{row['feature']!s:<24}  "
                f"adj.p={'-' if adjusted is None else f'{adjusted:.4g}'}  "
                f"effect={row['effect']:.4g}"
                + ("  withdrawn before C" if row["withdrawn"] else "")
            )
        print(f"{len(rows)} claim(s)")
        return command.EXIT_OK

    @staticmethod
    def run_show(args: argparse.Namespace) -> int:
        """Prints one claim with its hypothesis and checks.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        print(
            json.dumps(
                claims_audit.inspect(args.run_dir, args.claim_id),
                indent=2,
                sort_keys=True,
            )
        )
        return command.EXIT_OK

    @staticmethod
    def run_verify(args: argparse.Namespace) -> int:
        """Verifies the run's artifacts and prints each step.

        A failed step raises, which `main` turns into exit status 1.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0) if every step passed.

        Raises:
            ArtifactError: If any verification step failed.
        """
        key = (
            pathlib.Path(args.key_file).read_text(encoding="utf-8").strip()
            if args.key_file
            else None
        )
        result = claims_audit.verify(args.run_dir, key)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            for found in result.findings:
                print(f"{found.status.upper():<8}{found.name}: {found.detail}")
            print(
                f"{result.verified} of {len(result.findings)} steps passed"
                + ("" if result.ok else "; verification FAILED")
            )
        if not result.ok:
            raise errors.ArtifactError("claims verification failed")
        return command.EXIT_OK


class Trail(command.Command):
    """Builds, verifies and prints the trail of one deployed model."""

    name = "trail"
    help = "build, verify or show the business trail of a deployed model"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the `build`, `verify` and `show` actions.

        Args:
            parser: The argument parser of this command.
        """
        actions = parser.add_subparsers(dest="trail_action", required=True)
        build = actions.add_parser(
            "build", help="copy the retained records and seal the trail"
        )
        build.add_argument("output", help="trail folder to write")
        build.add_argument("--run", required=True, help="charter run folder")
        build.add_argument("--framing", required=True, help="frame output")
        build.add_argument("--registry", required=True, help="registry root")
        build.add_argument(
            "--deployment",
            required=True,
            help="loop folder with observed_outcomes.json",
        )
        build.add_argument("--inputs", required=True, help="signed inputs")
        verify = actions.add_parser(
            "verify", help="check a trail folder against its own records"
        )
        verify.add_argument("trail_dir")
        show = actions.add_parser("show", help="print the trail")
        show.add_argument("trail_dir")
        for sub in (build, verify):
            sub.add_argument(
                "--keys",
                help="JSON file mapping analyst, stakeholder and reviewer "
                "to keys; default: SCMLE_SIGNING_KEY for the analyst only",
            )
            sub.add_argument("--json", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).

        Raises:
            ArtifactError: If the built or verified trail does not verify.
        """
        if args.trail_action == "show":
            print(scmle.trail.show.render(args.trail_dir))
            return command.EXIT_OK
        keys = self.keys(args.keys)
        if args.trail_action == "build":
            report = builder.build(
                args.output,
                run_dir=args.run,
                framing_dir=args.framing,
                registry_dir=args.registry,
                deployment_dir=args.deployment,
                inputs_file=args.inputs,
                keys=keys,
            )
        else:
            report = chain.verify(args.trail_dir, keys)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            for found in report.findings:
                print(f"{found.kind:<30}{found.step}: {found.detail}")
            print(
                f"{report.steps} steps, head {report.head[:12]}; "
                + ("verified" if report.ok else "NOT verified")
            )
        if not report.ok:
            raise errors.ArtifactError("trail verification failed")
        return command.EXIT_OK

    @staticmethod
    def keys(path: str | None) -> dict[str, str | None]:
        """Returns the signing keys by role.

        The keys come from the file if one is given, else only the analyst's
        key comes from the environment.

        Args:
            path: A JSON file mapping role to key, or None.

        Returns:
            The keys by role; with no file, only the analyst's key from the
            environment (None if it is not set).
        """
        if path:
            loaded = json.loads(pathlib.Path(path).read_text("utf-8"))
            return {str(k): str(v) for k, v in loaded.items()}
        configured = signing.lookup(os.environ)
        return {"analyst": configured}
