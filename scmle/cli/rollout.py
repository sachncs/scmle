"""Commands that operate and deploy a release: operate and deploy."""

import argparse
import json
import logging
import pathlib

from scmle.cli import command
from scmle.contracts import errors
from scmle.deploy import gate
from scmle.deploy import livetests
from scmle.deploy import parity
from scmle.deploy import release
from scmle.deploy import release_state
import scmle.deploy.record
from scmle.governance import deployment_ledger
from scmle.governance import registry
import scmle.governance.envelope
import scmle.operations.controller

LOGGER = logging.getLogger(__name__)


class Operation(command.Command):
    """Runs the unattended controller: ticks, envelopes, status and halt."""

    name = "operate"
    help = (
        "run the unattended controller on an inbox; grant or revoke authority"
    )

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the controller actions.

        Args:
            parser: The argument parser of this command.
        """
        actions = parser.add_subparsers(dest="operate_action", required=True)
        tick = actions.add_parser(
            "tick", help="apply new inbox files once and step the loop"
        )
        tick.add_argument("deploy_dir")
        tick.add_argument(
            "--inbox",
            required=True,
            help="folder with batches/*.csv and labels/*.csv",
        )
        tick.add_argument(
            "--keys-file",
            help="JSON map from key id to the key that verifies it",
        )
        status = actions.add_parser("status", help="show the controller state")
        status.add_argument("deploy_dir")
        grant = actions.add_parser(
            "grant", help="record a person's signed envelope"
        )
        grant.add_argument("deploy_dir")
        grant.add_argument("--envelope", required=True, help="envelope JSON")
        grant.add_argument(
            "--key-file",
            required=True,
            help="file with the key that verifies it (the person's)",
        )
        revoke = actions.add_parser("revoke", help="remove an envelope")
        revoke.add_argument("deploy_dir")
        revoke.add_argument("envelope_id")
        revoke.add_argument("--reason", required=True)
        halt = actions.add_parser(
            "halt", help="stop the loop; every case is escalated"
        )
        halt.add_argument("deploy_dir")
        halt.add_argument("--reason", required=True)

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status of the chosen action.
        """
        action = getattr(self, f"run_{args.operate_action}")
        status: int = action(args)
        return status

    @staticmethod
    def open(
        deploy_dir: str, keys: dict[str, str] | None = None
    ) -> scmle.operations.controller.Controller:
        """Opens the controller of a deployment, mapping ledger faults.

        Args:
            deploy_dir: Directory of the deployment.
            keys: Verification keys by key id, or None.

        Returns:
            The controller.

        Raises:
            ArtifactError: If the deployment ledger cannot be opened.
        """
        try:
            return scmle.operations.controller.Controller(
                deploy_dir, verification_keys=keys
            )
        except deployment_ledger.MisuseError as error:
            raise errors.ArtifactError(
                f"cannot open the deployment: {error}"
            ) from error

    @classmethod
    def run_tick(cls, args: argparse.Namespace) -> int:
        """Applies the inbox once and prints what happened.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        keys = (
            json.loads(pathlib.Path(args.keys_file).read_text(encoding="utf-8"))
            if args.keys_file
            else None
        )
        result = cls.open(args.deploy_dir, keys).tick(args.inbox)
        print(json.dumps(result.to_dict()))
        return command.EXIT_OK

    @classmethod
    def run_status(cls, args: argparse.Namespace) -> int:
        """Prints the controller's state.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        print(json.dumps(cls.open(args.deploy_dir).status(), indent=2))
        return command.EXIT_OK

    @classmethod
    def run_grant(cls, args: argparse.Namespace) -> int:
        """Records an envelope after verifying its signature.

        The signature is verified under the key from the key file.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).

        Raises:
            ConfigError: If the key file cannot be read or holds no key, or
                the envelope is refused.
        """
        key = Deployment.read(args.key_file)
        envelope = deployment_ledger.Envelope.from_dict(
            json.loads(pathlib.Path(args.envelope).read_text(encoding="utf-8"))
        )
        controller = cls.open(
            args.deploy_dir, {deployment_ledger.thumbprint(key): key}
        )
        try:
            controller.envelopes.grant(envelope)
        except (
            scmle.governance.envelope.RefusedError,
            deployment_ledger.MisuseError,
        ) as error:
            raise errors.ConfigError(f"envelope refused: {error}") from error
        print(json.dumps(controller.status()["envelopes"]))
        return command.EXIT_OK

    @classmethod
    def run_revoke(cls, args: argparse.Namespace) -> int:
        """Removes an envelope's authority.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).

        Raises:
            ConfigError: If the revocation is refused.
        """
        controller = cls.open(args.deploy_dir)
        try:
            controller.envelopes.revoke(args.envelope_id, args.reason)
        except deployment_ledger.MisuseError as error:
            raise errors.ConfigError(f"revoke refused: {error}") from error
        print(json.dumps(controller.status()["envelopes"]))
        return command.EXIT_OK

    @classmethod
    def run_halt(cls, args: argparse.Namespace) -> int:
        """Halts the loop and prints the controller state.

        Nothing is recorded again if the loop is already halted. Only a
        recalibration that passes releases the halt.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        controller = cls.open(args.deploy_dir)
        controller.loop.suspend("operator_halt", {"reason": args.reason})
        controller.loop.save_state()
        print(json.dumps(controller.status()))
        return command.EXIT_OK


class Deployment(command.Command):
    """Release evidence: live checks, the promotion gate, deployment records."""

    name = "deploy"
    help = "live checks of a service, the promotion gate, deployment records"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Adds the deployment actions.

        The actions are `live`, `gate`, `model-digest`, `reconcile` and
        `verify-record`.

        Args:
            parser: The argument parser of this command.
        """
        actions = parser.add_subparsers(dest="deploy_action", required=True)
        live = actions.add_parser(
            "live", help="health, schema, inference and security checks"
        )
        live.add_argument("--url", required=True, help="base URL (https)")
        live.add_argument("--token-file", required=True)
        live.add_argument("--read-only-token-file")
        live.add_argument("--golden", default="deploy/local/golden")
        live.add_argument("--environment", required=True)
        live.add_argument("--commit", required=True)
        live.add_argument("--container-digest", required=True)
        live.add_argument("--model-digest", required=True)
        live.add_argument("--timeout", type=float, default=30.0)
        live.add_argument("--output", required=True, help="results JSON")
        decide = actions.add_parser(
            "gate", help="allow or refuse a release from its evidence"
        )
        decide.add_argument("stage", choices=["admit", "certify"])
        decide.add_argument("manifest", help="release manifest JSON")
        decide.add_argument("--key-file", required=True)
        decide.add_argument("--output", help="signed record (certify)")
        digest = actions.add_parser(
            "model-digest", help="digest a service reports for a version"
        )
        digest.add_argument("registry", help="registry directory")
        digest.add_argument("--model", required=True)
        digest.add_argument("--version", type=int, required=True)
        reconcile = actions.add_parser(
            "reconcile",
            help="compare the registry and ledger with the runtime report",
        )
        reconcile.add_argument("registry", help="registry directory")
        reconcile.add_argument("--model", required=True)
        reconcile.add_argument("--environment", required=True)
        reconcile.add_argument(
            "--observation",
            required=True,
            help="observation JSON (Observation.to_dict)",
        )
        reconcile.add_argument(
            "--ledger", help="deployment ledger file (deployment_ledger.jsonl)"
        )
        reconcile.add_argument("--output", help="write the report here")
        verify = actions.add_parser(
            "verify-record", help="check a deployment record's signature"
        )
        verify.add_argument("record")
        verify.add_argument("--key-file", required=True)

    def execute(self, args: argparse.Namespace) -> int:
        """Runs the chosen action.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status of the chosen action.
        """
        action = getattr(self, f"run_{args.deploy_action.replace('-', '_')}")
        status: int = action(args)
        return status

    @staticmethod
    def read(path: str) -> str:
        """Reads a key from a file (never from the command line).

        Args:
            path: The key file.

        Returns:
            The key, with surrounding whitespace removed.

        Raises:
            ConfigError: If the file cannot be read or holds no key.
        """
        try:
            key = pathlib.Path(path).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise errors.ConfigError(
                f"cannot read the key file {path}: {error.strerror}"
            ) from error
        if not key:
            raise errors.ConfigError(f"{path} holds no key")
        return key

    @staticmethod
    def run_live(args: argparse.Namespace) -> int:
        """Runs the four live checks and writes their results JSON.

        The checks are health, schema, inference and security. The results go
        to `--output` and a name-to-passed map is printed.

        Args:
            args: The parsed command-line arguments.

        Returns:
            EXIT_OK (0) if every check passed, else EXIT_FAILED (1).
        """
        read_only = (
            Deployment.read(args.read_only_token_file)
            if args.read_only_token_file
            else None
        )
        target = livetests.Target(
            environment=args.environment,
            source_commit=args.commit,
            container_digest=args.container_digest,
            model_digest=args.model_digest,
            token=Deployment.read(args.token_file),
            base_url=args.url,
            read_only_token=read_only,
        )
        golden = parity.goldens(args.golden)
        expected = parity.read_json(pathlib.Path(args.golden) / "expected.json")
        results = livetests.run_checks(
            livetests.http_fetch(args.url, args.timeout),
            target,
            golden,
            expected,
        )
        payload = [item.to_dict() for item in results]
        pathlib.Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps({item.name: item.passed for item in results}))
        return (
            command.EXIT_OK
            if all(item.passed for item in results)
            else command.EXIT_FAILED
        )

    @staticmethod
    def run_gate(args: argparse.Namespace) -> int:
        """Prints the gate decision; a refusal returns the blocked status.

        When the decision allows the release and carries a signed record, the
        record is written to `--output` if that is given.

        Args:
            args: The parsed command-line arguments.

        Returns:
            EXIT_BLOCKED (2) if the gate refuses, else EXIT_OK (0).
        """
        key = Deployment.read(args.key_file)
        candidate = release.gather(args.manifest)
        if args.stage == "admit":
            decision = gate.admit(candidate, key)
        else:
            decision = gate.certify(candidate, key)
        print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
        if not decision.allowed:
            return command.EXIT_BLOCKED
        if decision.record is not None and args.output:
            scmle.deploy.record.write_record(args.output, decision.record)
        return command.EXIT_OK

    @staticmethod
    def run_reconcile(args: argparse.Namespace) -> int:
        """Prints the reconciliation report; status 1 if the sources disagree.

        The report is also written to `--output` when that is given.

        Args:
            args: The parsed command-line arguments.

        Returns:
            EXIT_OK (0) if they agree, else EXIT_FAILED (1).
        """
        store = registry.ModelRegistryStore(args.registry)
        ledger = (
            deployment_ledger.DeploymentLedger.load(args.ledger)
            if args.ledger
            else None
        )
        seen = release.rebuild(
            release.read_json(pathlib.Path(args.observation))
        )
        report = release_state.ReleaseRegistry(store, ledger).reconcile(
            args.model, args.environment, seen
        )
        text = json.dumps(report.to_dict(), indent=2, sort_keys=True)
        if args.output:
            pathlib.Path(args.output).write_text(
                text + "\n", encoding="utf-8", newline="\n"
            )
        print(text)
        return command.EXIT_OK if report.consistent else command.EXIT_FAILED

    @staticmethod
    def run_model_digest(args: argparse.Namespace) -> int:
        """Prints the digest a service running this version reports.

        Args:
            args: The parsed command-line arguments.

        Returns:
            The exit status, EXIT_OK (0).
        """
        store = registry.ModelRegistryStore(args.registry)
        version = store.resolve(args.model, args.version)
        print(gate.model_digest_of(store, version))
        return command.EXIT_OK

    @staticmethod
    def run_verify_record(args: argparse.Namespace) -> int:
        """Prints whether the record's signature verifies, with its digest.

        Args:
            args: The parsed command-line arguments.

        Returns:
            EXIT_OK (0) if the signature verifies, else EXIT_FAILED (1).
        """
        record = scmle.deploy.record.read_record(args.record)
        valid = record.verify(Deployment.read(args.key_file))
        print(json.dumps({"valid": valid, "digest": record.digest()}))
        return command.EXIT_OK if valid else command.EXIT_FAILED
