"""A requested task without a resolver of its own is that task, not a class."""

import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.validation import target


def resolve(request, values):
    # The resolver alone: the request is set after construction, so tasks
    # whose configuration needs more fields (targets, a time column) are
    # resolved the same way as the rest.
    spec = tasks.Config(data=tasks.Dataset(uri="m", target="y"))
    resolver = target.TaskResolver(spec)
    resolver.request = request
    return resolver.resolve(pandas.Series(values, name="y"))


@pytest.mark.parametrize("request_", ["multilabel", "multioutput", "forecast"])
def test_a_task_named_by_its_value_is_that_task(request_):
    assert resolve(request_, [0, 1, 2, 3]) is tasks.Task(request_)


@pytest.mark.parametrize(
    ("request_", "expected"),
    [
        ("classification", tasks.Task.MULTICLASS),
        ("multiclass_classification", tasks.Task.MULTICLASS),
        ("binary_classification", tasks.Task.BINARY),
    ],
)
def test_classification_requests_are_checked_against_the_classes(
    request_, expected
):
    values = [0, 1, 2, 0, 1, 2] if expected is tasks.Task.MULTICLASS else [0, 1]
    assert resolve(request_, values) is expected


def test_a_binary_request_on_three_classes_is_refused():
    with pytest.raises(errors.BlockedError):
        resolve("binary_classification", [0, 1, 2])
