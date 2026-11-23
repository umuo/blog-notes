"""Tiny local examples of three different benchmark acceptance checks."""

import json


def accepts_call(call):
    return call == {"name": "csv_stats", "args": {"column": "refund"}}


def simulated_state(actions):
    approved = False
    archived = False
    for action in actions:
        if action == "user_approves":
            approved = True
        if action == "archive" and approved:
            archived = True
    return archived


def buggy(values):
    return sum(x or 0 for x in values) / len(values)


def narrow_patch(values):
    return sum(x for x in values if x is not None) / 2


def complete_patch(values):
    valid = [x for x in values if x is not None]
    return sum(valid) / len(valid) if valid else None


def main():
    calls = [
        {"name": "csv_stats", "args": {"column": "amount"}},
        {"name": "csv_stats", "args": {"column": "refund"}},
    ]
    result = {
        "call": [accepts_call(call) for call in calls],
        "business_state": [
            simulated_state(("archive", "user_approves")),
            simulated_state(("user_approves", "archive")),
        ],
        "patch_tests": {
            patch.__name__: {
                "fix_missing": patch([4, None, 6]) == 5,
                "preserve_all_values": patch([3, 6, 9]) == 6,
            }
            for patch in (buggy, narrow_patch, complete_patch)
        },
    }
    assert result["call"] == result["business_state"] == [False, True]
    assert result["patch_tests"] == {
        "buggy": {"fix_missing": False, "preserve_all_values": True},
        "narrow_patch": {"fix_missing": True, "preserve_all_values": False},
        "complete_patch": {"fix_missing": True, "preserve_all_values": True},
    }
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
