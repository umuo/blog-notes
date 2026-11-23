"""Compare sparse, milestone, and flawed rewards in a small report workflow."""

import json
from collections import defaultdict
from random import Random


ACTIONS = ("read", "write", "request", "wait", "archive", "announce")
LIMIT = 9


def step(state, action, response_delay):
    stage, waited = state
    if stage == 0 and action == "read":
        return (1, 0), True, False
    if stage == 1 and action == "write":
        return (2, 0), True, False
    if stage == 2 and action == "request":
        return (3, 0), True, False
    if stage == 3 and action == "wait":
        return ((4, 0) if waited + 1 >= response_delay else (3, waited + 1)), True, False
    if stage == 4 and action == "archive":
        return (5, 0), True, True
    return state, action == "announce", False


def episode(table, response_delay, rng, reward_mode, *, train):
    state = (0, 0)
    invalid = 0
    announcements = 0
    for move in range(1, LIMIT + 1):
        values = table[state[0]]
        best = max(values)
        if train:
            choices = [i for i, value in enumerate(values) if value == best]
            action_id = rng.randrange(len(ACTIONS)) if rng.random() < 0.35 else rng.choice(choices)
        else:
            action_id = values.index(best)
        action = ACTIONS[action_id]
        next_state, valid, completed = step(state, action, response_delay)
        invalid += int(not valid)
        announcements += int(action == "announce")
        reward = float(completed)
        if reward_mode != "terminal" and valid and next_state[0] > state[0] and not completed:
            reward += 0.15
        if reward_mode == "flawed" and action == "announce":
            reward += 0.5
        if train:
            future = 0 if completed or move == LIMIT else 0.9 * max(table[next_state[0]])
            values[action_id] += 0.35 * (reward + future - values[action_id])
        state = next_state
        if completed:
            return True, invalid, move, announcements
    return False, invalid, LIMIT, announcements


def main():
    summary = {}
    for mode in ("terminal", "milestones", "flawed"):
        held_out = []
        for seed in range(30):
            rng = Random(seed)
            table = defaultdict(lambda: [0.0] * len(ACTIONS))
            for _ in range(125):
                episode(table, rng.choice((2, 3)), rng, mode, train=True)
            held_out.append(episode(table, 4, rng, mode, train=False))
        summary[mode] = {
            "completed": sum(done for done, _, _, _ in held_out),
            "episodes": len(held_out),
            "invalid_actions": sum(bad for _, bad, _, _ in held_out),
            "announcements": sum(count for _, _, _, count in held_out),
            "steps": sum(moves for _, _, moves, _ in held_out),
        }
    assert summary["flawed"]["completed"] < summary["milestones"]["completed"]
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
