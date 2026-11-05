"""Static policy checks over the Terraform under ``deploy/terraform``.

The checks read the files with `scmle.deploy.hcl` and never run Terraform, so
they need no tool and no credentials. They answer one question, "does what is
written declare what the policy requires?", and nothing else: a pass is not
`terraform validate`, not a plan and not a statement about a deployed
account. Every rule returns `Violation` records with a file and line.

Rules:

``ingress``
    No security group, load balancer, route or IAM binding admits the public
    without authentication: no ``0.0.0.0/0`` ingress, a load balancer that is
    not internal needs an authenticating listener, every gateway route names
    an authorizer, tasks get no public address, and no ``allUsers`` or
    ``allAuthenticatedUsers`` binding exists.
``encryption``
    Buckets, registries, log groups, secrets, topics and keys are encrypted
    with a customer-managed key where the provider supports it, buckets block
    public access, image tags are immutable.
``iam``
    No wildcard action, no wildcard principal except a Deny, no wildcard
    resource except for the few APIs that cannot be scoped, no administrator
    or primitive role, and separate deploy, train and serve identities whose
    permissions do not overlap in the ways that matter.
``budget``
    A budget with notifications exists for each provider module and each
    environment sets a positive amount.
``image``
    The image is a digest, validated, never a tag.
``secrets``
    No credential-shaped text, no secret-named variable default, no secret in
    an ``environment`` list, no state or key file.
``limits``
    Task or instance resources, autoscaling bounds, private networking.
``monitoring``
    Alarms or alert policies exist.
``environments``
    dev, staging and production exist per provider, name themselves, label
    their resources with ``scmle-environment`` and keep the backend free of
    credentials.
"""

from collections.abc import Callable
import dataclasses
import pathlib
import re

from scmle.deploy import contract
from scmle.deploy import hcl

PROVIDERS = ("aws", "gcp")
ENVIRONMENTS = ("dev", "staging", "production")
MODULES = {"aws": "aws_environment", "gcp": "gcp_environment"}
LABEL_KEY = "scmle-environment"
STAR_RESOURCE_ACTIONS = frozenset(
    {
        "ecr:GetAuthorizationToken",
        "ecs:DescribeTaskDefinition",
        "ecs:RegisterTaskDefinition",
    }
)
"""Actions with no resource-level permission: ``*`` is the only resource."""
PUBLIC_CIDRS = ("0.0.0.0/0", "::/0")
PRIMITIVE_ROLES = ("roles/owner", "roles/editor", "roles/viewer")
PUBLIC_MEMBERS = ("allUsers", "allAuthenticatedUsers")
ADMIN_POLICIES = ("AdministratorAccess", "PowerUserAccess", "FullAccess")
SECRET_TEXT = (
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY"),
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}"),
)
SECRET_ATTRIBUTES = (
    "password",
    "secret_string",
    "secret_data",
    "client_secret",
    "private_key",
    "access_key",
    "secret_key",
    "token",
    "credentials",
)
FORBIDDEN_FILES = ("*.tfstate", "*.tfstate.*", "*.tfvars", "*.pem", "*.key")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
IMAGE_ATTRIBUTE = re.compile(r'\bimage\s*=\s*("(?:[^"\\]|\\.)*")')
ENV_PAIR = re.compile(
    r'name\s*=\s*"([A-Za-z0-9_]+)"\s*,?\s*value\s*=\s*"([^"]*)"'
)
WRITE_ACTIONS = ("s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion")


@dataclasses.dataclass(frozen=True, slots=True)
class Violation:
    """One broken rule at one place."""

    rule: str
    location: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Returns a JSON-ready record."""
        return {
            "rule": self.rule,
            "location": self.location,
            "message": self.message,
        }


@dataclasses.dataclass(slots=True)
class Unit:
    """The blocks of one directory (a module or an environment)."""

    path: pathlib.Path
    provider: str
    kind: str
    blocks: list[hcl.Block]

    def resources(self, *types: str) -> list[hcl.Block]:
        """Resource blocks of the given types."""
        return [
            b
            for b in self.blocks
            if b.kind == "resource" and b.labels and b.labels[0] in types
        ]

    def data(self, *types: str) -> list[hcl.Block]:
        """Data blocks of the given types."""
        return [
            b
            for b in self.blocks
            if b.kind == "data" and b.labels and b.labels[0] in types
        ]


def where(block: hcl.Block) -> str:
    """``file:line address`` for a block."""
    return f"{block.path}:{block.line} {'.'.join(block.labels) or block.kind}"


def strings(text: str | None) -> list[str]:
    """The quoted strings in an attribute text."""
    return hcl.QUOTED.findall(text or "")


def string_items(text: str | None) -> list[str]:
    """The string elements of a list attribute."""
    value = hcl.literal(text)
    if isinstance(value, list):
        return [str(item) for item in value]
    return strings(text)


def flag(
    out: list[Violation], rule: str, block: hcl.Block, message: str
) -> None:
    """Appends a violation located at ``block``."""
    out.append(Violation(rule, where(block), message))


def load(root: pathlib.Path) -> list[Unit]:
    """Reads every directory of ``root`` that holds ``.tf`` files."""
    units: list[Unit] = []
    for provider in PROVIDERS:
        module = root / "modules" / MODULES[provider]
        if module.is_dir():
            units.append(
                Unit(module, provider, "module", hcl.parse_directory(module))
            )
        for env in ENVIRONMENTS:
            directory = root / provider / "environments" / env
            if directory.is_dir():
                units.append(
                    Unit(
                        directory,
                        provider,
                        env,
                        hcl.parse_directory(directory),
                    )
                )
    return units


def check_ingress(unit: Unit) -> list[Violation]:
    """No public entrance without authentication."""
    out: list[Violation] = []
    for block in unit.blocks:
        for item in block.walk():
            for name in ("member", "members"):
                if any(
                    word in strings(item.attributes.get(name))
                    for word in PUBLIC_MEMBERS
                ):
                    flag(out, "ingress", block, "allUsers binding")
    for ingress_block in unit.resources(
        "aws_vpc_security_group_ingress_rule", "aws_security_group_rule"
    ):
        attributes = ingress_block.attributes
        kind = hcl.literal(attributes.get("type"))
        if kind in (None, "ingress") and any(
            cidr in strings(attributes.get(name))
            for name in ("cidr_ipv4", "cidr_ipv6", "cidr_blocks")
            for cidr in PUBLIC_CIDRS
        ):
            flag(out, "ingress", ingress_block, "public ingress CIDR")
    for group in unit.resources("aws_security_group"):
        for rule in group.child("ingress"):
            if any(
                cidr in strings(rule.attributes.get("cidr_blocks"))
                for cidr in PUBLIC_CIDRS
            ):
                flag(out, "ingress", group, "public ingress CIDR")
    out.extend(check_load_balancers(unit))
    for route in unit.resources("aws_apigatewayv2_route"):
        kind = hcl.literal(route.attributes.get("authorization_type"))
        if kind not in ("JWT", "AWS_IAM", "CUSTOM"):
            flag(out, "ingress", route, "route without an authorizer")
        elif kind != "AWS_IAM" and "authorizer_id" not in route.attributes:
            flag(out, "ingress", route, "route names no authorizer")
    for service in unit.resources("aws_ecs_service"):
        for network in service.child("network_configuration"):
            if hcl.literal(network.attributes.get("assign_public_ip")) is not (
                False
            ):
                flag(out, "ingress", service, "tasks may get a public address")
    for run in unit.resources("google_cloud_run_v2_service"):
        if hcl.literal(run.attributes.get("invoker_iam_disabled")) is True:
            flag(out, "ingress", run, "invoker IAM check disabled")
    if unit.resources("google_cloud_run_v2_service") and not any(
        hcl.literal(b.attributes.get("role")) == "roles/run.invoker"
        for b in unit.resources("google_cloud_run_v2_service_iam_member")
    ):
        flag(
            out,
            "ingress",
            unit.resources("google_cloud_run_v2_service")[0],
            "no roles/run.invoker binding: callers cannot be authenticated",
        )
    for variable in unit.blocks if unit.kind == "module" else []:
        if variable.kind == "variable" and variable.labels == (
            "invoker_members",
        ):
            text = " ".join(
                c.attributes.get("condition", "")
                for c in variable.child("validation")
            )
            if "allUsers" not in text:
                flag(
                    out,
                    "ingress",
                    variable,
                    "invoker_members validation does not refuse allUsers",
                )
    return out


def check_load_balancers(unit: Unit) -> list[Violation]:
    """A load balancer that is not internal must authenticate."""
    out: list[Violation] = []
    listeners = unit.resources("aws_lb_listener")
    for balancer in unit.resources("aws_lb"):
        if hcl.literal(balancer.attributes.get("internal")) is True:
            continue
        for listener in listeners:
            actions = listener.child("default_action")
            authenticated = any(
                hcl.literal(a.attributes.get("type")) == "authenticate-oidc"
                for a in actions
            )
            if (
                hcl.literal(listener.attributes.get("protocol")) != "HTTPS"
                or not authenticated
            ):
                flag(
                    out,
                    "ingress",
                    listener,
                    "listener of a public load balancer neither is HTTPS "
                    "nor authenticates",
                )
        if not listeners:
            flag(out, "ingress", balancer, "public load balancer")
    return out


def mentions(text: str | None, *words: str) -> bool:
    """Whether the attribute text contains any of ``words``."""
    return any(word in (text or "") for word in words)


def check_encryption(unit: Unit) -> list[Violation]:
    """Encryption at rest, public-access blocks, immutable tags."""
    out: list[Violation] = []
    encrypted = {
        ref(b.attributes.get("bucket"))
        for b in unit.resources(
            "aws_s3_bucket_server_side_encryption_configuration"
        )
        if mentions(walk_attributes(b), "aws:kms")
    }
    blocked = {
        ref(b.attributes.get("bucket"))
        for b in unit.resources("aws_s3_bucket_public_access_block")
        if all(
            hcl.literal(b.attributes.get(n)) is True
            for n in (
                "block_public_acls",
                "block_public_policy",
                "ignore_public_acls",
                "restrict_public_buckets",
            )
        )
    }
    for bucket in unit.resources("aws_s3_bucket"):
        name = bucket.labels[1]
        if name not in encrypted:
            flag(out, "encryption", bucket, "bucket not encrypted with KMS")
        if name not in blocked:
            flag(out, "encryption", bucket, "public access not blocked")
    for repo in unit.resources("aws_ecr_repository"):
        if hcl.literal(repo.attributes.get("image_tag_mutability")) != (
            "IMMUTABLE"
        ):
            flag(out, "encryption", repo, "image tags are mutable")
        crypt = repo.child("encryption_configuration")
        if not crypt or hcl.literal(
            crypt[0].attributes.get("encryption_type")
        ) != ("KMS"):
            flag(out, "encryption", repo, "repository not encrypted with KMS")
        scan = repo.child("image_scanning_configuration")
        if not scan or hcl.literal(
            scan[0].attributes.get("scan_on_push")
        ) is not (True):
            flag(out, "encryption", repo, "scan on push is off")
    required = {
        "aws_cloudwatch_log_group": "kms_key_id",
        "aws_secretsmanager_secret": "kms_key_id",  # nosec B105
        "aws_sns_topic": "kms_master_key_id",
    }
    for kind, attribute in required.items():
        for block in unit.resources(kind):
            if attribute not in block.attributes:
                flag(out, "encryption", block, f"no {attribute}")
    for log in unit.resources("aws_cloudwatch_log_group"):
        if "retention_in_days" not in log.attributes:
            flag(out, "encryption", log, "no log retention")
    for key in unit.resources("aws_kms_key"):
        if hcl.literal(key.attributes.get("enable_key_rotation")) is not True:
            flag(out, "encryption", key, "key rotation is off")
    for bucket in unit.resources("google_storage_bucket"):
        enc = bucket.child("encryption")
        if not enc or "default_kms_key_name" not in enc[0].attributes:
            flag(out, "encryption", bucket, "bucket has no customer key")
        if hcl.literal(bucket.attributes.get("public_access_prevention")) != (
            "enforced"
        ):
            flag(out, "encryption", bucket, "public access not prevented")
        if (
            hcl.literal(bucket.attributes.get("uniform_bucket_level_access"))
            is not True
        ):
            flag(out, "encryption", bucket, "uniform access is off")
    for registry in unit.resources("google_artifact_registry_repository"):
        config = registry.child("docker_config")
        if not config or hcl.literal(
            config[0].attributes.get("immutable_tags")
        ) is not (True):
            flag(out, "encryption", registry, "image tags are mutable")
    for key in unit.resources("google_kms_crypto_key"):
        if "rotation_period" not in key.attributes:
            flag(out, "encryption", key, "key rotation is off")
    return out


def ref(text: str | None) -> str:
    """The resource name in a reference like ``aws_s3_bucket.data.id``."""
    parts = (text or "").strip().split(".")
    return parts[1] if len(parts) > 2 else ""


def walk_attributes(block: hcl.Block) -> str:
    """All attribute texts of a block and its descendants, joined."""
    return " ".join(
        text for item in block.walk() for text in item.attributes.values()
    )


def statements(unit: Unit) -> list[tuple[hcl.Block, hcl.Block]]:
    """Every (policy document, statement) pair."""
    pairs = []
    for document in unit.data("aws_iam_policy_document"):
        for statement in document.child("statement"):
            pairs.append((document, statement))
    return pairs


def key_policy_documents(unit: Unit) -> set[str]:
    """Names of documents attached to a KMS key (resource ``*`` is the key)."""
    names = set()
    for key in unit.resources("aws_kms_key"):
        match = re.search(
            r"data\.aws_iam_policy_document\.(\w+)",
            key.attributes.get("policy", ""),
        )
        if match:
            names.add(match.group(1))
    return names


def check_iam(unit: Unit) -> list[Violation]:
    """Least privilege and separate identities."""
    out: list[Violation] = []
    key_documents = key_policy_documents(unit)
    actions_of: dict[str, set[str]] = {}
    for document, statement in statements(unit):
        actions = set(string_items(statement.attributes.get("actions")))
        actions_of.setdefault(document.labels[1], set()).update(actions)
        effect = hcl.literal(statement.attributes.get("effect")) or "Allow"
        for action in sorted(actions):
            if "*" in action:
                flag(out, "iam", statement, f"wildcard action {action}")
        resources = string_items(statement.attributes.get("resources"))
        if (
            "*" in resources
            and document.labels[1] not in key_documents
            and not actions <= STAR_RESOURCE_ACTIONS
        ):
            flag(out, "iam", statement, "wildcard resource")
        for principal in statement.child("principals"):
            identifiers = string_items(principal.attributes.get("identifiers"))
            if "*" in identifiers and effect != "Deny":
                flag(out, "iam", statement, "wildcard principal")
    for trust in unit.data("aws_iam_policy_document"):
        for statement in trust.child("statement"):
            if "sts:AssumeRoleWithWebIdentity" in string_items(
                statement.attributes.get("actions")
            ):
                tests = [
                    hcl.literal(c.attributes.get("test"))
                    for c in statement.child("condition")
                ]
                variables = " ".join(
                    c.attributes.get("variable", "")
                    for c in statement.child("condition")
                )
                if "StringLike" in tests or ":sub" not in variables:
                    flag(
                        out,
                        "iam",
                        statement,
                        "federated trust must match the subject exactly",
                    )
    for attachment in unit.resources(
        "aws_iam_role_policy_attachment", "aws_iam_policy_attachment"
    ):
        if mentions(attachment.attributes.get("policy_arn"), *ADMIN_POLICIES):
            flag(out, "iam", attachment, "administrator or full-access policy")
    for inline in unit.resources("aws_iam_role_policy", "aws_iam_policy"):
        if '"*"' in inline.attributes.get("policy", "") and "jsonencode" in (
            inline.attributes.get("policy", "")
        ):
            flag(out, "iam", inline, "inline policy with a wildcard")
    for binding in unit.resources(
        "google_project_iam_member",
        "google_project_iam_binding",
        "google_storage_bucket_iam_member",
        "google_secret_manager_secret_iam_member",
        "google_artifact_registry_repository_iam_member",
        "google_cloud_run_v2_service_iam_member",
        "google_service_account_iam_member",
    ):
        role = hcl.literal(binding.attributes.get("role")) or ""
        if not isinstance(role, str):
            continue
        if (
            role in PRIMITIVE_ROLES
            or role.endswith("admin")
            or role.endswith("Admin")
        ):
            flag(out, "iam", binding, f"primitive or admin role {role}")
        if binding.labels[0].startswith("google_project_iam"):
            flag(out, "iam", binding, "project-wide grant")
    out.extend(check_separation(unit, actions_of))
    return out


def check_separation(
    unit: Unit, actions_of: dict[str, set[str]]
) -> list[Violation]:
    """Deploy, train and serve are separate identities with separate powers."""
    out: list[Violation] = []
    if unit.provider == "aws" and unit.kind == "module":
        roles = {b.labels[1] for b in unit.resources("aws_iam_role")}
        missing = {"deploy", "train", "serve"} - roles
        if missing:
            out.append(
                Violation(
                    "iam",
                    str(unit.path),
                    "missing roles: " + ", ".join(sorted(missing)),
                )
            )
        serve = actions_of.get("serve", set())
        if serve & set(WRITE_ACTIONS):
            out.append(
                Violation("iam", str(unit.path), "serve role can write data")
            )
        deploy = actions_of.get("deploy", set())
        if any(
            a.startswith(("s3:", "secretsmanager:", "kms:")) for a in deploy
        ):
            out.append(
                Violation(
                    "iam",
                    str(unit.path),
                    "deploy role can touch data, secrets or keys",
                )
            )
        train = actions_of.get("train", set())
        if any(
            a.startswith(("iam:", "ecs:", "secretsmanager:")) for a in train
        ):
            out.append(
                Violation(
                    "iam",
                    str(unit.path),
                    "train role has deployment or secret powers",
                )
            )
    if unit.provider == "gcp" and unit.kind == "module":
        accounts = {
            b.labels[1] for b in unit.resources("google_service_account")
        }
        missing = {"deploy", "train", "serve"} - accounts
        if missing:
            out.append(
                Violation(
                    "iam",
                    str(unit.path),
                    "missing service accounts: " + ", ".join(sorted(missing)),
                )
            )
    return out


def check_budget(unit: Unit) -> list[Violation]:
    """A budget with notifications in each module; an amount per environment."""
    out: list[Violation] = []
    if unit.kind == "module":
        for budget in unit.resources("aws_budgets_budget"):
            if not budget.child("notification"):
                flag(out, "budget", budget, "budget has no notification")
        for budget in unit.resources("google_billing_budget"):
            if not budget.child("threshold_rules"):
                flag(out, "budget", budget, "budget has no threshold rules")
        if not unit.resources("aws_budgets_budget", "google_billing_budget"):
            out.append(
                Violation("budget", str(unit.path), "no budget declared")
            )
        return out
    for module in (b for b in unit.blocks if b.kind == "module"):
        amount = hcl.literal(module.attributes.get("monthly_budget_usd"))
        if (
            isinstance(amount, bool)
            or not isinstance(amount, int | float)
            or amount <= 0
        ):
            flag(out, "budget", module, "monthly_budget_usd is not positive")
    return out


def image_problem(text: str) -> str | None:
    """Why an image reference is not a digest, or None if it is."""
    reference = text.strip().strip('"')
    if ":latest" in reference:
        return "image uses the latest tag"
    if "@" not in reference:
        return "image is not pinned by digest"
    tail = reference.split("@", 1)[1]
    if "image_digest" in tail or DIGEST.fullmatch(tail):
        return None
    return "image digest is neither the validated variable nor a sha256"


def check_image(unit: Unit) -> list[Violation]:
    """Images are digests; the digest variable validates its shape."""
    out: list[Violation] = []
    for block in unit.blocks:
        for item in block.walk():
            for name, text in item.attributes.items():
                if name == "image" or name == "container_definitions":
                    pairs = (
                        IMAGE_ATTRIBUTE.findall(text)
                        if name == "container_definitions"
                        else [text]
                    )
                    for pair in pairs:
                        problem = image_problem(pair)
                        if problem:
                            flag(out, "image", block, problem)
    if unit.kind == "module":
        variable = [
            b
            for b in unit.blocks
            if b.kind == "variable" and b.labels == ("image_digest",)
        ]
        if not variable:
            out.append(Violation("image", str(unit.path), "no image_digest"))
        for item in variable:
            text = " ".join(
                c.attributes.get("condition", "")
                for c in item.child("validation")
            )
            if "sha256:" not in text or "[0-9a-f]{64}" not in text:
                flag(out, "image", item, "image_digest does not validate")
            if "default" in item.attributes:
                flag(out, "image", item, "image_digest has a default")
    for module in (b for b in unit.blocks if b.kind == "module"):
        value = module.attributes.get("image_digest", "")
        if value.strip() != "var.image_digest" and not DIGEST.fullmatch(
            value.strip().strip('"')
        ):
            flag(out, "image", module, "image_digest is not a digest")
    return out


def check_secrets(unit: Unit) -> list[Violation]:
    """No secret values in the files."""
    out: list[Violation] = []
    for path in sorted(unit.path.rglob("*")):
        if path.is_file() and any(path.match(p) for p in FORBIDDEN_FILES):
            out.append(Violation("secrets", str(path), "state or key file"))
        if path.suffix == ".tf":
            text = path.read_text(encoding="utf-8")
            for pattern in SECRET_TEXT:
                if pattern.search(text):
                    out.append(
                        Violation(
                            "secrets", str(path), "credential-shaped text"
                        )
                    )
            for name, value in ENV_PAIR.findall(text):
                if contract.looks_secret(name) and value:
                    out.append(
                        Violation(
                            "secrets", str(path), f"secret value in env {name}"
                        )
                    )
    for block in unit.blocks:
        for item in block.walk():
            for name, text in item.attributes.items():
                if name in SECRET_ATTRIBUTES and hcl.literal(text):
                    flag(out, "secrets", block, f"literal value for {name}")
        if block.kind == "variable" and "default" in block.attributes:
            name = block.labels[0].upper()
            safe = name.endswith(("_NAMES", "_IDS", "_ARNS"))
            if contract.looks_secret(name) and not safe:
                flag(out, "secrets", block, "secret-named variable default")
        if block.kind == "terraform":
            for backend in block.child("backend"):
                for name in backend.attributes:
                    flag(
                        out,
                        "secrets",
                        block,
                        f"backend attribute {name} is set in the file",
                    )
    return out


def check_limits(unit: Unit) -> list[Violation]:
    """Resource limits, autoscaling bounds and private networking."""
    out: list[Violation] = []
    if unit.kind != "module":
        return out
    for task in unit.resources("aws_ecs_task_definition"):
        for name in ("cpu", "memory"):
            if name not in task.attributes:
                flag(out, "limits", task, f"no {name}")
        text = task.attributes.get("container_definitions", "")
        if "readonlyRootFilesystem = true" not in text:
            flag(out, "limits", task, "root filesystem is writable")
        if not re.search(r'\buser\s*=\s*"[1-9][0-9]*', text):
            flag(out, "limits", task, "container does not run as non-root")
        if "secrets" not in text or "valueFrom" not in text:
            flag(out, "limits", task, "secrets are not references")
    if unit.provider == "aws":
        targets = unit.resources("aws_appautoscaling_target")
        if not targets or not unit.resources("aws_appautoscaling_policy"):
            out.append(Violation("limits", str(unit.path), "no autoscaling"))
        for target in targets:
            for name in ("min_capacity", "max_capacity"):
                if name not in target.attributes:
                    flag(out, "limits", target, f"no {name}")
        if not unit.resources("aws_vpc"):
            out.append(Violation("limits", str(unit.path), "no VPC"))
        for subnet in unit.resources("aws_subnet"):
            if hcl.literal(
                subnet.attributes.get("map_public_ip_on_launch")
            ) is not (False):
                flag(out, "limits", subnet, "subnet hands out public IPs")
        for service in unit.resources("aws_ecs_service"):
            if not service.child("deployment_circuit_breaker"):
                flag(out, "limits", service, "no deployment circuit breaker")
    for run in unit.resources("google_cloud_run_v2_service"):
        for template in run.child("template"):
            scaling = template.child("scaling")
            if not scaling or "max_instance_count" not in scaling[0].attributes:
                flag(out, "limits", run, "no max_instance_count")
            if "max_instance_request_concurrency" not in template.attributes:
                flag(out, "limits", run, "no request concurrency")
            if not template.child("vpc_access"):
                flag(out, "limits", run, "no private network access")
            for container in template.child("containers"):
                limits = [
                    r
                    for r in container.child("resources")
                    if "limits" in r.attributes
                ]
                if not limits:
                    flag(out, "limits", run, "container has no limits")
                for env in container.child("env"):
                    env_name = hcl.literal(env.attributes.get("name"))
                    if (
                        isinstance(env_name, str)
                        and contract.looks_secret(env_name)
                        and "value" in env.attributes
                    ):
                        flag(out, "secrets", run, f"plain value for {env_name}")
    for subnet in unit.resources("google_compute_subnetwork"):
        if hcl.literal(
            subnet.attributes.get("private_ip_google_access")
        ) is not (True):
            flag(out, "limits", subnet, "private Google access is off")
    return out


def check_monitoring(unit: Unit) -> list[Violation]:
    """Alarms or alert policies exist."""
    out: list[Violation] = []
    if unit.kind != "module":
        return out
    alarms = unit.resources(
        "aws_cloudwatch_metric_alarm", "google_monitoring_alert_policy"
    )
    if len(alarms) < 2:
        out.append(
            Violation("monitoring", str(unit.path), "fewer than 2 alerts")
        )
    return out


def check_environment(unit: Unit) -> list[Violation]:
    """Environments name themselves, label resources and hold no credentials."""
    out: list[Violation] = []
    if unit.kind == "module":
        if unit.provider == "aws":
            for local in (b for b in unit.blocks if b.kind == "locals"):
                if LABEL_KEY in local.attributes.get("tags", ""):
                    break
            else:
                out.append(
                    Violation("environments", str(unit.path), "no tag locals")
                )
        else:
            for kind in (
                "google_storage_bucket",
                "google_artifact_registry_repository",
                "google_secret_manager_secret",
                "google_cloud_run_v2_service",
            ):
                for block in unit.resources(kind):
                    if "labels" not in block.attributes:
                        flag(out, "environments", block, "no labels")
            if not any(
                LABEL_KEY in b.attributes.get("labels", "")
                for b in unit.blocks
                if b.kind == "locals"
            ):
                out.append(
                    Violation("environments", str(unit.path), "no label locals")
                )
        return out
    modules = [b for b in unit.blocks if b.kind == "module"]
    if len(modules) != 1:
        out.append(
            Violation("environments", str(unit.path), "expected one module")
        )
        return out
    module = modules[0]
    if hcl.literal(module.attributes.get("environment")) != unit.kind:
        flag(out, "environments", module, f"environment is not {unit.kind}")
    expected = f"../../../modules/{MODULES[unit.provider]}"
    if hcl.literal(module.attributes.get("source")) != expected:
        flag(out, "environments", module, "module source is not the shared one")
    if not any(
        b.kind == "terraform" and b.child("backend") for b in unit.blocks
    ):
        out.append(Violation("environments", str(unit.path), "no backend"))
    if unit.provider == "aws":
        text = " ".join(
            walk_attributes(c)
            for b in unit.blocks
            if b.kind == "provider"
            for c in b.child("default_tags")
        )
        if LABEL_KEY not in text or f'"{unit.kind}"' not in text:
            out.append(
                Violation(
                    "environments",
                    str(unit.path),
                    "provider default_tags lack the environment tag",
                )
            )
    if unit.kind == "production":
        floor = hcl.literal(
            module.attributes.get("min_tasks")
            or module.attributes.get("min_instances")
        )
        if not isinstance(floor, int) or floor < 2:
            flag(out, "environments", module, "production runs fewer than 2")
        if hcl.literal(module.attributes.get("destroyable")) is not False:
            flag(out, "environments", module, "production is destroyable")
        if (
            hcl.literal(module.attributes.get("deletion_protection"))
            is not True
        ):
            flag(out, "environments", module, "production has no protection")
    return out


CHECKS: tuple[Callable[[Unit], list[Violation]], ...] = (
    check_ingress,
    check_encryption,
    check_iam,
    check_budget,
    check_image,
    check_secrets,
    check_limits,
    check_monitoring,
    check_environment,
)


def check_tree(root: str | pathlib.Path) -> list[Violation]:
    """Runs every rule over every module and environment under ``root``.

    A missing provider, module or environment is itself a violation.

    Raises:
        HclError: If a file is outside the reader's subset.
    """
    base = pathlib.Path(root)
    units = load(base)
    out: list[Violation] = []
    for provider in PROVIDERS:
        if not (base / "modules" / MODULES[provider]).is_dir():
            out.append(
                Violation("environments", str(base), f"no {provider} module")
            )
        for env in ENVIRONMENTS:
            if not (base / provider / "environments" / env).is_dir():
                out.append(
                    Violation("environments", str(base), f"no {provider} {env}")
                )
    for unit in units:
        for check in CHECKS:
            out.extend(check(unit))
    return sorted(set(out), key=lambda v: (v.rule, v.location, v.message))
