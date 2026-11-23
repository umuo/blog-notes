"""Compare a fixed refund metric query with frozen SQL candidates."""

import csv
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory


SCHEMA = """
CREATE TABLE refund(id TEXT PRIMARY KEY, day TEXT NOT NULL, cents INTEGER);
CREATE TABLE refund_tag(id TEXT NOT NULL, tag TEXT NOT NULL);
"""
CANONICAL = "SELECT COUNT(cents), AVG(cents) FROM refund WHERE day >= ? AND day < ?"
RANGE = ("2026-11-01", "2026-12-01")
TABLES = {"refund": {"id", "day", "cents", ""}, "refund_tag": {"id", "tag", ""}}
HERE = Path(__file__).resolve().parent
FUNCTIONS = {"avg", "coalesce", "count"}


def authorize(action, first, second, _database, _trigger):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and second in TABLES.get(first, set()):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and second.lower() in FUNCTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def query(database, sql, parameters=(), *, budget=1000):
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as reader:
        reader.set_authorizer(authorize)
        remaining = budget

        def consume():
            nonlocal remaining
            remaining -= 1
            return remaining < 0

        reader.set_progress_handler(consume, 100)
        try:
            rows = reader.execute(sql, parameters).fetchmany(11)
        except sqlite3.DatabaseError as exc:
            return {"error": exc.sqlite_errorname}
        return {"rows": rows} if len(rows) <= 10 else {"error": "row_limit"}


def main():
    with TemporaryDirectory() as root:
        database = Path(root) / "example.db"
        with closing(sqlite3.connect(database)) as writer:
            writer.executescript(SCHEMA)
            with (HERE / "refunds.csv").open(newline="") as source:
                rows = [(row["id"], row["day"], None if row["cents"] == "NA" else int(row["cents"]))
                        for row in csv.DictReader(source)]
                writer.executemany("INSERT INTO refund VALUES (?, ?, ?)", rows)
            writer.executemany("INSERT INTO refund_tag VALUES (?, ?)", [("r1", "new"), ("r1", "vip"), ("r3", "new")])
            writer.commit()
        scenarios = {
            "bound_canonical": query(database, CANONICAL, RANGE),
            "wrong_join": query(database, "SELECT COUNT(cents), AVG(cents) FROM refund JOIN refund_tag USING (id) WHERE day >= ? AND day < ?", RANGE),
            "null_as_zero": query(database, "SELECT COUNT(*), AVG(COALESCE(cents, 0)) FROM refund WHERE day >= ? AND day < ?", RANGE),
            "inclusive_end": query(database, "SELECT COUNT(cents), AVG(cents) FROM refund WHERE day BETWEEN ? AND ?", RANGE),
            "write_rejected": query(database, "UPDATE refund SET cents = 0"),
            "budget_rejected": query(database, "SELECT COUNT(*) FROM refund a, refund b, refund c, refund d, refund e", budget=0),
        }
        assert scenarios["bound_canonical"] == {"rows": [(2, 500.0)]}
        assert scenarios["wrong_join"]["rows"][0][0] == 3
        assert scenarios["write_rejected"]["error"] in {"SQLITE_AUTH", "SQLITE_READONLY"}
        assert scenarios["budget_rejected"] == {"error": "SQLITE_INTERRUPT"}
        print(json.dumps({"metric_source": "dictionary/v2/p1#amount + dictionary/v2/p1#missing",
                          "range": RANGE, "unit": "cents", "scenarios": scenarios}, indent=4))


if __name__ == "__main__":
    main()
