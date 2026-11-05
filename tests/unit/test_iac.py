"""The Terraform policy check, its reader and the destroy-orphan checker."""

import json
import pathlib
import shutil
import subprocess

import pytest

from scmle.contracts import errors
from scmle.deploy import hcl
from scmle.deploy import iac_policy
from scmle.deploy import orphans
from scripts import check_terraform_policy

ROOT = pathlib.Path(__file__).resolve().parents[2]
TREE = ROOT / "deploy" / "terraform"
AWS = "modules/aws_environment"
GCP = "modules/gcp_environment"


@pytest.fixture
def tree(tmp_path):
    target = tmp_path / "terraform"
    shutil.copytree(TREE, target)
    return target


def rules(root):
    return {violation.rule for violation in iac_policy.check_tree(root)}


def messages(root):
    return [v.message for v in iac_policy.check_tree(root)]


def edit(root, relative, old, new):
    path = root / relative
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {relative}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_the_committed_tree_satisfies_every_rule():
    assert iac_policy.check_tree(TREE) == []


def test_three_environments_per_provider_exist():
    for provider in ("aws", "gcp"):
        for env in ("dev", "staging", "production"):
            assert (
                TREE / provider / "environments" / env / "main.tf"
            ).is_file()


def test_the_reader_reads_blocks_attributes_and_nesting():
    blocks = hcl.parse(
        'resource "t" "n" {\n  a = "x"\n  l = [\n    "p",\n    "q",\n  ]\n'
        "  # note\n  inner {\n    b = 1\n  }\n}\n"
    )
    block = blocks[0]
    assert block.address == "t.n"
    assert hcl.literal(block.attributes["a"]) == "x"
    assert hcl.literal(block.attributes["l"]) == ["p", "q"]
    assert hcl.literal(block.child("inner")[0].attributes["b"]) == 1


def test_the_reader_does_not_evaluate_interpolations():
    assert hcl.literal('"a-${var.x}"') is None
    assert hcl.literal("var.x") is None


@pytest.mark.parametrize(
    "text",
    ['resource "t" "n" {\n a = <<EOT\nx\nEOT\n}', 'resource "t" {', "x = 1"],
)
def test_the_reader_refuses_what_it_cannot_read(text):
    with pytest.raises(hcl.HclError):
        hcl.parse(text)


def test_public_ingress_cidr_is_refused(tree):
    edit(
        tree,
        f"{AWS}/network.tf",
        'resource "aws_vpc_security_group_ingress_rule" "task_from_alb" {\n'
        "  security_group_id            = aws_security_group.task.id\n"
        "  referenced_security_group_id = aws_security_group.alb.id\n",
        'resource "aws_vpc_security_group_ingress_rule" "task_from_alb" {\n'
        "  security_group_id            = aws_security_group.task.id\n"
        '  cidr_ipv4 = "0.0.0.0/0"\n',
    )
    assert "public ingress CIDR" in messages(tree)


def test_a_route_without_an_authorizer_is_refused(tree):
    edit(
        tree,
        f"{AWS}/service.tf",
        'authorization_type = "JWT"',
        'authorization_type = "NONE"',
    )
    assert "route without an authorizer" in messages(tree)


def test_a_public_load_balancer_without_authentication_is_refused(tree):
    edit(
        tree,
        f"{AWS}/service.tf",
        "internal                   = true",
        "internal                   = false",
    )
    assert any(
        "public load balancer" in m or "authenticate" in m
        for m in messages(tree)
    )


def test_tasks_with_public_addresses_are_refused(tree):
    edit(
        tree,
        f"{AWS}/service.tf",
        "assign_public_ip = false",
        "assign_public_ip = true",
    )
    assert "tasks may get a public address" in messages(tree)


def test_all_users_is_refused(tree):
    edit(
        tree,
        f"{GCP}/iam.tf",
        "member   = each.value",
        'member   = "allUsers"',
    )
    assert "allUsers binding" in messages(tree)


def test_a_service_without_an_invoker_binding_is_refused(tree):
    edit(tree, f"{GCP}/iam.tf", '"roles/run.invoker"', '"roles/run.viewer"')
    assert any("roles/run.invoker" in m for m in messages(tree))


def test_unencrypted_storage_is_refused(tree):
    edit(tree, f"{AWS}/storage.tf", '"aws:kms"', '"AES256"')
    assert "bucket not encrypted with KMS" in messages(tree)
    edit(tree, f"{AWS}/storage.tf", '"IMMUTABLE"', '"MUTABLE"')
    assert "image tags are mutable" in messages(tree)


def test_a_bucket_without_a_customer_key_is_refused(tree):
    edit(tree, f"{GCP}/main.tf", "default_kms_key_name", "ignored_name")
    assert "bucket has no customer key" in messages(tree)


def test_wildcards_in_iam_are_refused(tree):
    edit(
        tree,
        f"{AWS}/iam.tf",
        'actions   = ["s3:GetObject"]',
        'actions   = ["s3:*"]',
    )
    assert any(m.startswith("wildcard action") for m in messages(tree))


def test_a_wildcard_resource_is_refused_unless_the_api_cannot_be_scoped(tree):
    edit(
        tree,
        f"{AWS}/iam.tf",
        'resources = [aws_ecr_repository.this.arn]\n  }\n\n  statement {\n    sid       = "WriteLogs"',
        'resources = ["*"]\n  }\n\n  statement {\n    sid       = "WriteLogs"',
    )
    assert "wildcard resource" in messages(tree)


def test_an_administrator_role_is_refused(tree):
    edit(
        tree,
        f"{GCP}/iam.tf",
        '"roles/run.developer"',
        '"roles/owner"',
    )
    assert any("primitive or admin role" in m for m in messages(tree))


def test_serve_may_not_write_and_deploy_may_not_touch_data(tree):
    edit(
        tree,
        f"{AWS}/iam.tf",
        'sid       = "ReadBundles"\n    actions   = ["s3:GetObject"]',
        'sid       = "ReadBundles"\n    actions   = ["s3:GetObject", "s3:PutObject"]',
    )
    assert "serve role can write data" in messages(tree)


def test_federated_trust_must_match_the_subject_exactly(tree):
    edit(
        tree,
        f"{AWS}/iam.tf",
        'test     = "StringEquals"\n      variable = "token.actions.githubusercontent.com:sub"',
        'test     = "StringLike"\n      variable = "token.actions.githubusercontent.com:sub"',
    )
    assert any("subject exactly" in m for m in messages(tree))


def test_a_missing_budget_is_refused(tree):
    edit(
        tree,
        "aws/environments/staging/main.tf",
        "monthly_budget_usd  = 300",
        "monthly_budget_usd  = 0",
    )
    assert "monthly_budget_usd is not positive" in messages(tree)
    shutil.rmtree(tree / GCP)
    shutil.copytree(TREE / GCP, tree / GCP)
    text = (tree / GCP / "monitoring.tf").read_text()
    start = text.index('resource "google_billing_budget"')
    (tree / GCP / "monitoring.tf").write_text(text[:start])
    assert "no budget declared" in messages(tree)


def test_a_tagged_image_is_refused(tree):
    edit(tree, f"{AWS}/service.tf", "@${var.image_digest}", ":latest")
    assert "image uses the latest tag" in messages(tree)
    edit(tree, f"{GCP}/service.tf", "@${var.image_digest}", ":v1")
    assert "image is not pinned by digest" in messages(tree)


def test_an_environment_must_pass_a_digest_not_a_tag(tree):
    edit(tree, "gcp/environments/dev/main.tf", "var.image_digest", '"v1"')
    assert "image_digest is not a digest" in messages(tree)


def test_an_unvalidated_digest_variable_is_refused(tree):
    edit(tree, f"{AWS}/variables.tf", "[0-9a-f]{64}", "[0-9a-f]+")
    assert "image_digest does not validate" in messages(tree)


def test_credential_text_and_secret_literals_are_refused(tree):
    edit(
        tree,
        f"{AWS}/network.tf",
        "locals {",
        "# AKIAABCDEFGHIJKLMNOP\nlocals {",
    )
    assert "credential-shaped text" in messages(tree)


def test_a_secret_literal_and_a_secret_default_are_refused(tree):
    with (tree / AWS / "extra.tf").open("w", encoding="utf-8") as handle:
        handle.write(
            'resource "aws_secretsmanager_secret_version" "v" {\n'
            '  secret_string = "hunter2"\n}\n\n'
            'variable "api_token" {\n  default = "abc"\n}\n'
        )
    found = messages(tree)
    assert "literal value for secret_string" in found
    assert "secret-named variable default" in found


def test_state_and_variable_files_are_refused(tree):
    (tree / AWS / "terraform.tfstate").write_text("{}", encoding="utf-8")
    assert "state or key file" in messages(tree)


def test_a_backend_with_settings_in_the_file_is_refused(tree):
    edit(
        tree,
        "aws/environments/dev/main.tf",
        'backend "s3" {}',
        'backend "s3" {\n    bucket = "x"\n  }',
    )
    assert any("backend attribute bucket" in m for m in messages(tree))


def test_missing_limits_and_autoscaling_are_refused(tree):
    edit(tree, f"{AWS}/service.tf", "max_capacity       = var.max_tasks\n", "")
    assert "no max_capacity" in messages(tree)
    edit(
        tree,
        f"{GCP}/service.tf",
        "max_instance_count = var.max_instances",
        "ignored = 1",
    )
    assert "no max_instance_count" in messages(tree)


def test_a_root_filesystem_that_is_writable_is_refused(tree):
    edit(
        tree,
        f"{AWS}/service.tf",
        "readonlyRootFilesystem = true",
        "readonlyRootFilesystem = false",
    )
    assert "root filesystem is writable" in messages(tree)


def test_a_plain_secret_env_value_is_refused(tree):
    edit(
        tree,
        f"{GCP}/service.tf",
        '      env {\n        name  = "SCMLE_AUTH_ISSUER"\n        value = var.jwt_issuer\n      }\n',
        '      env {\n        name  = "SCMLE_AUTH_TOKEN"\n        value = "x"\n      }\n',
    )
    assert any("plain value for SCMLE_AUTH_TOKEN" in m for m in messages(tree))


def test_production_must_run_two_and_be_protected(tree):
    edit(
        tree,
        "aws/environments/production/main.tf",
        "min_tasks           = 2",
        "min_tasks           = 1",
    )
    edit(
        tree,
        "gcp/environments/production/main.tf",
        "deletion_protection = true",
        "deletion_protection = false",
    )
    found = messages(tree)
    assert "production runs fewer than 2" in found
    assert "production has no protection" in found


def test_an_environment_must_name_itself(tree):
    edit(
        tree,
        "gcp/environments/staging/main.tf",
        'environment       = "staging"',
        'environment       = "dev"',
    )
    assert "environment is not staging" in messages(tree)


def test_a_missing_environment_is_a_violation(tree):
    shutil.rmtree(tree / "gcp" / "environments" / "staging")
    assert "no gcp staging" in messages(tree)


def test_no_alert_policy_is_refused(tree):
    text = (tree / AWS / "monitoring.tf").read_text()
    start = text.index('resource "aws_cloudwatch_metric_alarm" "service_cpu"')
    end = text.index('resource "aws_budgets_budget"')
    (tree / AWS / "monitoring.tf").write_text(text[:start] + text[end:])
    assert "fewer than 2 alerts" in messages(tree)


def test_the_policy_script_reports_and_exits_nonzero(tree, capsys):
    assert check_terraform_policy.main([str(tree)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    edit(tree, f"{AWS}/storage.tf", '"IMMUTABLE"', '"MUTABLE"')
    assert check_terraform_policy.main([str(tree)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert report["violations"][0]["rule"] == "encryption"


def test_the_tool_check_skips_when_no_terraform_is_installed(tmp_path):
    bash = shutil.which("bash")
    assert bash is not None
    done = subprocess.run(
        [bash, str(ROOT / "scripts" / "terraform_checks.sh")],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0
    assert "skipped" in done.stderr


# ---- the destroy-orphan checker ------------------------------------------

AWS_LISTING = {
    "ResourceTagMappingList": [
        {
            "ResourceARN": "arn:aws:s3:::scmle-dev-111-data",
            "Tags": [{"Key": "scmle-environment", "Value": "dev"}],
        },
        {
            "ResourceARN": "arn:aws:kms:eu-west-1:111:key/abc",
            "Tags": [{"Key": "scmle-environment", "Value": "dev"}],
            "State": "PendingDeletion",
        },
        {
            "ResourceARN": "arn:aws:s3:::scmle-staging-111-data",
            "Tags": [{"Key": "scmle-environment", "Value": "staging"}],
        },
        {
            "ResourceARN": "arn:aws:logs:eu-west-1:111:log-group:/scmle/scmle-dev/serve",
            "Tags": [],
        },
        {"ResourceARN": "arn:aws:s3:::unrelated", "Tags": []},
    ]
}


def test_orphans_are_listed_from_a_seeded_aws_listing():
    report = orphans.find_orphans("aws", "dev", AWS_LISTING)
    assert not report.clean
    assert [o.identifier for o in report.orphans] == [
        "arn:aws:s3:::scmle-dev-111-data",
        "arn:aws:logs:eu-west-1:111:log-group:/scmle/scmle-dev/serve",
    ]
    assert "untagged" in report.orphans[1].reason
    assert [p.identifier for p in report.pending] == [
        "arn:aws:kms:eu-west-1:111:key/abc"
    ]


def test_another_environment_and_unrelated_resources_are_not_orphans():
    report = orphans.find_orphans("aws", "production", AWS_LISTING)
    assert report.clean and not report.pending


def test_a_clean_destroy_with_only_pending_deletions_is_clean():
    listing = {
        "ResourceTagMappingList": [AWS_LISTING["ResourceTagMappingList"][1]]
    }
    report = orphans.find_orphans("aws", "dev", listing)
    assert report.clean and len(report.pending) == 1


def test_resources_left_in_the_state_are_reported():
    report = orphans.find_orphans(
        "aws",
        "dev",
        {"ResourceTagMappingList": []},
        ["module.environment.aws_vpc.this\n", ""],
    )
    assert not report.clean
    assert report.state == ("module.environment.aws_vpc.this",)


def test_gcp_labels_and_states_are_read():
    listing = [
        {
            "name": "//storage.googleapis.com/scmle-dev-1-data",
            "assetType": "storage.googleapis.com/Bucket",
            "labels": {"scmle-environment": "dev"},
            "state": "ACTIVE",
        },
        {
            "name": "//run.googleapis.com/projects/p/services/scmle-dev-serve",
            "assetType": "run.googleapis.com/Service",
            "labels": {"scmle-environment": "dev"},
            "state": "DELETE_REQUESTED",
        },
    ]
    report = orphans.find_orphans("gcp", "dev", listing)
    assert [o.kind for o in report.orphans] == ["storage.googleapis.com/Bucket"]
    assert len(report.pending) == 1


def test_an_unreadable_listing_is_an_error_not_a_clean_result():
    with pytest.raises(errors.ArtifactError):
        orphans.find_orphans("aws", "dev", {"unexpected": []})
    with pytest.raises(errors.ArtifactError):
        orphans.find_orphans("azure", "dev", [])
    with pytest.raises(errors.ArtifactError):
        orphans.find_orphans("gcp", "dev", {"a": 1})


def test_the_orphan_script_exit_status(tmp_path, capsys):
    listing = tmp_path / "listing.json"
    listing.write_text(json.dumps(AWS_LISTING), encoding="utf-8")
    base = [
        "--provider",
        "aws",
        "--environment",
        "dev",
        "--listing",
        str(listing),
    ]
    assert orphans.main(base) == 1
    assert json.loads(capsys.readouterr().out)["clean"] is False
    assert orphans.main([*base[:-1], str(tmp_path / "missing.json")]) == 2
    assert (
        orphans.main(
            [
                "--provider",
                "aws",
                "--environment",
                "production",
                "--listing",
                str(listing),
            ]
        )
        == 0
    )
