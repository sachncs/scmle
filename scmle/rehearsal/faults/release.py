"""The bad-release scenario, on a control plane made of local processes."""

from collections.abc import Mapping
import http
import json
import pathlib
import time
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.deploy import control
from scmle.deploy import observation
from scmle.deploy import parity
from scmle.deploy import release_state
from scmle.governance import deployment_ledger
from scmle.governance import registry
from scmle.operations import classify
from scmle.operations import recovery
from scmle.operations import signals
from scmle.rehearsal import demo
from scmle.rehearsal.faults import harness


class Processes(control.ControlPlane):
    """Revisions are local service processes; the interface is the same.

    ``deploy(digest)`` starts a revision of the folder registered for the
    digest and leaves earlier ones running, as a managed platform keeps
    earlier revisions; ``set_traffic`` points the single entry at one of
    them; ``observe`` reports the revision that holds the traffic, with its
    readiness from ``/readyz``.
    """

    provider = "local"

    def __init__(self, fleet: harness.Fleet) -> None:
        """Starts with no images and no revisions.

        Args:
            fleet: The fleet that owns the processes.
        """
        self.fleet = fleet
        self.images: dict[str, pathlib.Path] = {}
        self.revisions: dict[str, harness.Instance] = {}
        self.digests: dict[str, str] = {}
        self.primary: str | None = None
        self.count = 0

    def enroll(self, folder: pathlib.Path) -> str:
        """Starts a folder's service once to learn its digest and registers it.

        The service is started, its ``/health`` ``artifact_digest`` is read, and
        the process is killed; the digest then maps to the folder for `deploy`.

        Args:
            folder: A service folder.

        Returns:
            The artifact digest the service reports.

        Raises:
            ScmleError: If the service does not become ready.
        """
        probe = self.fleet.start(folder)
        if harness.boot(probe) is None:
            raise errors.ScmleError("the image does not start")
        digest = str(
            demo.retrieve(f"{probe.base}/health")[1]["artifact_digest"]
        )
        probe.process.kill()
        probe.process.wait(timeout=15)
        self.images[digest] = folder
        return digest

    def deploy(self, image_digest: str) -> str:
        """Starts a revision of an image; the first one gets the traffic.

        Earlier revisions are left running.

        Args:
            image_digest: The digest of a registered image.

        Returns:
            The name of the new revision, ``local-<n>``.
        """
        self.count += 1
        revision = f"local-{self.count}"
        self.revisions[revision] = self.fleet.start(self.images[image_digest])
        self.digests[revision] = image_digest
        if self.primary is None:
            self.primary = revision
        return revision

    def set_traffic(self, allocation: Mapping[str, int]) -> None:
        """Points all traffic at one revision.

        Args:
            allocation: A mapping from revision to percentage.

        Raises:
            Unsupported: If the allocation is not 100 percent to a single
                revision.
        """
        if (
            len(allocation) != 1
            or sum(allocation.values()) != harness.FULL_TRAFFIC_PERCENT
        ):
            raise control.Unsupported("only all traffic to one revision")
        self.primary = next(iter(allocation))

    def ready(self, revision: str) -> bool:
        """Whether a revision answers /readyz with 200.

        Args:
            revision: The revision to ask.

        Returns:
            True if it answers with status 200; False otherwise.
        """
        try:
            return (
                demo.retrieve(f"{self.revisions[revision].base}/readyz", 2.0)[0]
                == http.HTTPStatus.OK
            )
        except OSError:
            # handled: documented False when not ready
            return False

    def observe(self) -> observation.Observation:
        """What is running and where traffic goes.

        Returns:
            An observation of the revision that holds the traffic, with its
            image digest, its readiness (from ``/readyz``) and the sorted
            (revision, digest) pairs of every revision deployed.

        Raises:
            ArtifactError: If nothing is deployed.
        """
        if self.primary is None:
            raise errors.ArtifactError("nothing is deployed")
        return observation.Observation(
            provider=self.provider,
            service="local-service",
            revision=self.primary,
            image_digest=self.digests[self.primary],
            ready=self.ready(self.primary),
            traffic=(observation.Traffic(self.primary, 100),),
            observed_at=demo.now(),
            revisions=tuple(sorted(self.digests.items())),
        )

    def base(self) -> str:
        """URL of the revision that holds the traffic.

        Returns:
            The base URL of the primary revision.

        Raises:
            ArtifactError: If nothing is deployed.
        """
        if self.primary is None:
            raise errors.ArtifactError("nothing is deployed")
        return self.revisions[self.primary].base


def register(
    store: registry.ModelRegistryStore,
    release: release_state.ReleaseRegistry,
    spec: dict[str, Any],
    work: pathlib.Path,
    label: str,
) -> registry.Release:
    """Registers and approves one signed bundle.

    Args:
        store: The model registry.
        release: The release registry.
        spec: The reference model's specification.
        work: Directory in which the bundle is saved, under `label`.
        label: Name of the bundle, also used as the run id.

    Returns:
        The record of the registered version.
    """
    exported = parity.handwrite(spec).save(work / label, demo.DEMO_SIGNING_KEY)
    record = store.register(
        harness.MODEL,
        exported,
        metrics={},
        run_id=label,
        signing_key=demo.DEMO_SIGNING_KEY,
    )
    release.approve(
        harness.MODEL,
        record.version,
        demo.sanction(record, f"recovery run {label}", "recovery-run"),
        demo.DEMO_SIGNING_KEY,
    )
    return record


def degradation(
    plane: Processes,
    watch: signals.SignalMonitor,
    rng: numpy.random.Generator,
    known_good: list[Any] | None,
) -> tuple[int | None, float | None]:
    """Feeds noisy delayed labels to the monitor until it flags the release.

    Args:
        plane: The process plane whose service is asked for predictions.
        watch: The monitor that receives the labelled outcomes.
        rng: Source of the label noise.
        known_good: The answers of the good release, which stand in for the
            truth; labelling stops being possible without them.

    Returns:
        The number of labelled cases served when the loss was first classified
        as model performance and the `time.monotonic()` of that moment, or
        `(None, None)` if it never was.
    """
    for step in range(60):
        served = harness.predict(plane.base(), 3)[1].get("predictions") or []
        if not known_good or len(served) != len(known_good):
            # A request that failed or came back short has no labels to
            # compare with; it must not abort the run with a mismatch.
            continue
        truth = [
            p if rng.random() > harness.LABEL_NOISE_RATE else 1 - p
            for p in known_good
        ]
        watch.observe_outcomes(float(step), signals.misses(served, truth))
        verdict = classify.classify(watch.evaluate(float(step)))
        if verdict.verdict is classify.Verdict.MODEL_PERFORMANCE:
            return (step + 1) * 3, time.monotonic()
    return None, None


def assemble(
    fleet: harness.Fleet, base: pathlib.Path
) -> tuple[
    release_state.ReleaseRegistry,
    Processes,
    list[registry.Release],
    list[str],
]:
    """Registers three versions and enrolls their images on a process plane.

    The versions are the golden model, its inverted (wrong) copy and a third
    copy of the golden model, each signed, registered and approved.

    Args:
        fleet: The fleet that owns the processes.
        base: The scratch folder; the model registry lives in ``registry``.

    Returns:
        The release registry (over a fresh deployment ledger, with
        time-scaled polling), the plane, the three registered versions and
        their image digests, in that order.
    """
    store = registry.ModelRegistryStore(base / "registry")
    ledger = deployment_ledger.DeploymentLedger.create(
        None, charter_hash="recovery", policy={}, alpha_promo="1/20"
    )
    release = release_state.ReleaseRegistry(
        store,
        ledger,
        control.Polling(attempts=40, delay=recovery.scaled(0.5)),
    )
    good_spec = harness.specify()
    third_spec = {**good_spec, "model_version": "golden-3"}
    versions = [
        register(store, release, good_spec, base, "v1"),
        register(store, release, harness.invert(), base, "v2"),
        register(store, release, third_spec, base, "v3"),
    ]
    plane = Processes(fleet)
    images = [
        plane.enroll(harness.export_service(base / "img1", good_spec)),
        plane.enroll(harness.export_service(base / "img2", harness.invert())),
        plane.enroll(harness.export_service(base / "img3", third_spec)),
    ]
    return release, plane, versions, images


def promote(
    release: release_state.ReleaseRegistry,
    plane: Processes,
    record: registry.Release,
    image: str,
) -> release_state.Outcome:
    """Promotes a registered version of the recovery model to production.

    Args:
        release: The release registry.
        plane: The process plane that runs the image.
        record: The registered version.
        image: The digest of the version's image.

    Returns:
        The release registry's outcome.
    """
    return release.promote(
        harness.MODEL,
        record.version,
        plane,
        harness.ENVIRONMENT,
        image,
        signing_key=demo.DEMO_SIGNING_KEY,
    )


def recover(
    release: release_state.ReleaseRegistry,
    plane: Processes,
    known_good: list[Any] | None,
) -> tuple[int | None, float | None, release_state.Outcome, Any, bool]:
    """Detects the bad release from delayed labels and rolls it back.

    The delayed labels are the good model's answers with 5% noise.

    Args:
        release: The release registry.
        plane: The process plane serving the bad release.
        known_good: The answers of the good release.

    Returns:
        The labelled cases served at detection (or None), the seconds from
        detection to the confirmed rollback (or None), the rollback outcome,
        the predictions served afterwards, and whether registry, ledger and
        runtime agree.
    """
    rng = numpy.random.default_rng(3)
    watch = signals.SignalMonitor(
        signals.Window(window_seconds=1000.0, min_samples=30),
        signals.SignalThresholds(reference_loss=0.05, loss_tolerance=0.05),
    )
    detected_after, detected_at = degradation(plane, watch, rng, known_good)
    rolled = release.rollback(
        harness.MODEL,
        harness.ENVIRONMENT,
        plane,
        signing_key=demo.DEMO_SIGNING_KEY,
    )
    elapsed = time.monotonic() - detected_at if detected_at else None
    restored = harness.predict(plane.base())[1].get("predictions")
    consistent = release.reconcile(
        harness.MODEL, harness.ENVIRONMENT, plane.observe()
    ).consistent
    return detected_after, elapsed, rolled, restored, consistent


def withhold(
    release: release_state.ReleaseRegistry,
    plane: Processes,
    record: registry.Release,
    image: str,
    base: pathlib.Path,
) -> tuple[release_state.Outcome, bool, Any]:
    """Tries to promote a version whose image is broken so it never readies.

    Args:
        release: The release registry.
        plane: The process plane; the image's folder is replaced by a
            tampered copy.
        record: The registered version.
        image: The digest of the version's image.
        base: The scratch folder that receives the tampered copy.

    Returns:
        The promotion outcome, whether the release registry read the same
        before and after it, and the predictions served afterwards.
    """
    plane.images[image] = harness.corrupt(
        plane.images[image], base / "img3-broken"
    )
    before = json.dumps(release.read(), sort_keys=True)
    unready = promote(release, plane, record, image)
    after = json.dumps(release.read(), sort_keys=True)
    still = harness.predict(plane.base())[1].get("predictions")
    return unready, before == after, still


def bad_release() -> recovery.Outcome:
    """Promotes a wrong release, detects it from labels and rolls it back.

    A good and an inverted-coefficient version are promoted in turn. Noisy
    labels taken from the good model's answers are fed to a signal monitor
    until the loss is classified as model performance, then the release is
    rolled back and the registry, ledger and runtime are reconciled. A third
    release whose image cannot become ready must then be refused, leave the
    registry unchanged and leave the previous revision serving.

    Returns:
        The recovery result of this fault: its checks, its RTO and RPO
        measurements and the details observed.
    """
    with harness.stage("release", "bad_release", "local processes") as (
        fleet,
        base,
        result,
    ):
        release, plane, versions, images = assemble(fleet, base)
        first = promote(release, plane, versions[0], images[0])
        known_good = harness.predict(plane.base())[1].get("predictions")
        promoted = promote(release, plane, versions[1], images[1])
        wrong = harness.predict(plane.base())[1].get("predictions")
        detected_after, elapsed, rolled, restored, consistent = recover(
            release, plane, known_good
        )
        unready, unchanged, still = withhold(
            release, plane, versions[2], images[2], base
        )
        result.checks = {
            "the first and the bad release were promoted": first.done
            and promoted.done,
            "the known good release answered the labelling requests": bool(
                known_good
            ),
            "the bad release served different answers": wrong != known_good,
            "the loss signal classified it as model performance": (
                detected_after is not None
            ),
            "the rollback was confirmed by the platform": rolled.done,
            "the rolled-back service answers as before": restored == known_good,
            "registry, ledger and runtime agree after the rollback": consistent,
            "a release that never becomes ready was not promoted": (
                not unready.done and unready.restored is True
            ),
            "the refused release changed nothing in the registry": unchanged,
            "the previous revision still serves": still == known_good,
        }
        result.measurements = [
            recovery.Measurement(
                "service",
                elapsed,
                0,
                "from the labelled-outcome window classifying the release as "
                "model performance to the platform confirming the previous "
                "revision with all traffic (rollback and read-back). Detection "
                "delay is reported separately. " + harness.METHOD_CLOCK,
            )
        ]
        result.detail = {
            "detected_after_labelled_cases": detected_after,
            "rollback": rolled.to_dict(),
            "unready_release": unready.to_dict(),
            "control_plane": "Processes: revisions are local processes",
        }
    return result
