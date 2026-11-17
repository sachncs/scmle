"""The local demo: train, evaluate, package, predict, monitor, roll back.

One command (`make demo`, or `python -m scmle.rehearsal.demo run`) runs the
whole lifecycle on the explicit files in `deploy/local/`, with no cloud account:

1. train a good model and a bad release from two datasets, and evaluate them;
2. package the good model as a hardened service folder, store it in the mock
   object store, fetch it back by digest and serve it with the production
   rules (signed bundle, signed provenance, no `SCMLE_TRUSTED`);
3. predict through the service and the batch path and compare both with the
   bundle, then check the golden records;
4. show that a corrupted copy and an unsigned copy never become ready;
5. register both models, promote the bad release, monitor it on fresh
   labelled data, and roll back to the good one (the rollback drill);
6. ship metrics to the mock metrics sink and measure startup and latency.

`prepare` and `verify` split the same work for the compose file, where the
service runs in a container and the test runner calls it over the network.
The demo signing key in `fixture` is a published fixture, not a secret.

The package splits the demo: `fixture` (the published key, data, thresholds
and training), `journal` (`Transcript`), `client` (processes, HTTP, the store
and the sink), `stages` (the recorded steps) and `command` (`run`, `verify`
and `main`).
"""

from scmle.rehearsal.demo.client import authenticate
from scmle.rehearsal.demo.client import Authenticator
from scmle.rehearsal.demo.client import download
from scmle.rehearsal.demo.client import footprint
from scmle.rehearsal.demo.client import halt
from scmle.rehearsal.demo.client import launch
from scmle.rehearsal.demo.client import measure
from scmle.rehearsal.demo.client import post_json
from scmle.rehearsal.demo.client import push
from scmle.rehearsal.demo.client import retrieve
from scmle.rehearsal.demo.client import serialize
from scmle.rehearsal.demo.client import settle
from scmle.rehearsal.demo.client import sync
from scmle.rehearsal.demo.client import unready
from scmle.rehearsal.demo.client import unused
from scmle.rehearsal.demo.command import main
from scmle.rehearsal.demo.command import run
from scmle.rehearsal.demo.command import verify
from scmle.rehearsal.demo.fixture import DATA_DIRECTORY
from scmle.rehearsal.demo.fixture import DEMO_SIGNING_KEY
from scmle.rehearsal.demo.fixture import EXPECTED_SAMPLES
from scmle.rehearsal.demo.fixture import FAST_OPTIONS
from scmle.rehearsal.demo.fixture import GOLDEN_DIRECTORY
from scmle.rehearsal.demo.fixture import LATENCY_REQUESTS
from scmle.rehearsal.demo.fixture import MINIMUM_ROC_AUC
from scmle.rehearsal.demo.fixture import MODEL
from scmle.rehearsal.demo.fixture import now
from scmle.rehearsal.demo.fixture import PROBABILITY_TOLERANCE
from scmle.rehearsal.demo.fixture import PROMOTED_VERSION
from scmle.rehearsal.demo.fixture import READY_TIMEOUT_SECONDS
from scmle.rehearsal.demo.fixture import REVIEWER
from scmle.rehearsal.demo.fixture import sanction
from scmle.rehearsal.demo.fixture import TARGET
from scmle.rehearsal.demo.fixture import train
from scmle.rehearsal.demo.journal import Step
from scmle.rehearsal.demo.journal import Transcript
from scmle.rehearsal.demo.stages import compare
from scmle.rehearsal.demo.stages import corroborate
from scmle.rehearsal.demo.stages import drill
from scmle.rehearsal.demo.stages import instrument
from scmle.rehearsal.demo.stages import mirror
from scmle.rehearsal.demo.stages import prepare
from scmle.rehearsal.demo.stages import rollback
from scmle.rehearsal.demo.stages import tamper

__all__ = [
    "DATA_DIRECTORY",
    "DEMO_SIGNING_KEY",
    "EXPECTED_SAMPLES",
    "FAST_OPTIONS",
    "GOLDEN_DIRECTORY",
    "LATENCY_REQUESTS",
    "MINIMUM_ROC_AUC",
    "MODEL",
    "PROBABILITY_TOLERANCE",
    "PROMOTED_VERSION",
    "READY_TIMEOUT_SECONDS",
    "REVIEWER",
    "TARGET",
    "Authenticator",
    "Step",
    "Transcript",
    "authenticate",
    "compare",
    "corroborate",
    "download",
    "drill",
    "footprint",
    "halt",
    "instrument",
    "launch",
    "main",
    "measure",
    "mirror",
    "now",
    "post_json",
    "prepare",
    "push",
    "retrieve",
    "rollback",
    "run",
    "sanction",
    "serialize",
    "settle",
    "sync",
    "tamper",
    "train",
    "unready",
    "unused",
    "verify",
]
