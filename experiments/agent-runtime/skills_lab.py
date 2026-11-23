"""Discover one fixture skill through Pi's loader and Codex app-server."""

import argparse
import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


NAME = "refund-report-fixture"
ROOT = Path(__file__).parent
FIXTURE = ROOT / "skills" / NAME


async def codex(project):
    app = await asyncio.create_subprocess_exec(
        "codex", "app-server", "--listen", "stdio://", cwd=project,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def send(method, params=None, id_=None):
        body = {"method": method}
        if params is not None:
            body["params"] = params
        if id_ is not None:
            body["id"] = id_
        app.stdin.write(json.dumps(body).encode() + b"\n")
        await app.stdin.drain()
        if id_ is None:
            return None
        async with asyncio.timeout(30):
            while line := await app.stdout.readline():
                reply = json.loads(line)
                if reply.get("id") == id_:
                    return reply
        raise RuntimeError(f"Codex stdio closed before response for {method}")

    try:
        init = await send("initialize", {"clientInfo": {"name": "skill-fixture", "version": "1.0.0"}}, 1)
        assert "result" in init
        await send("initialized")
        answer = await send("skills/list", {"cwds": [str(project)], "forceReload": True}, 2)
        candidates = [skill for group in answer["result"]["data"] for skill in group["skills"]
                      if skill["name"] == NAME and Path(skill["path"]).is_relative_to(project)]
        assert len(candidates) == 1 and candidates[0]["enabled"]
        return {"discovered": True, "scope": candidates[0]["scope"],
                "version": subprocess.check_output(["codex", "--version"], text=True).strip()}
    finally:
        app.stdin.close()
        try:
            await asyncio.wait_for(app.wait(), timeout=3)
        except TimeoutError:
            app.terminate()
            await app.wait()


def pi(launcher, source, project):
    script = Path(project) / "list-skills.ts"
    module = (source / "packages/coding-agent/src/core/skills.ts").resolve().as_posix()
    script.write_text(
        f"import {{ loadSkillsFromDir }} from {json.dumps(module)};\n"
        "const result = loadSkillsFromDir({dir: process.argv[2], source: 'project'});\n"
        "console.log(JSON.stringify(result.skills.map(({name,description}) => ({name,description}))));\n"
    )
    raw = subprocess.check_output(
        [str(launcher), str(script), str(project / ".agents" / "skills")], text=True, timeout=30,
    )
    selected, = [skill for skill in json.loads(raw) if skill["name"] == NAME]
    return {"discovered": True, "version": "0.70.0", "description_matches": "refund CSV" in selected["description"]}


async def main(args):
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory).resolve()
        shutil.copytree(FIXTURE, project / ".agents" / "skills" / NAME)
        subprocess.run(["git", "init", "-q", str(project)], check=True, timeout=8)
        result = {"name": NAME, "manifest_version": "1.0.0",
                  "hosts": {"pi": pi(args.tsx, args.pi_source, project), "codex": await codex(project)},
                  "automatic_model_selection_tested": False}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsx", type=Path, required=True)
    parser.add_argument("--pi-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
