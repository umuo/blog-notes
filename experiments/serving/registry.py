"""Register two validated release manifests and exercise a rollback in SQLite."""

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
NAME = "refund-json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.dir.absolute()
    folder.mkdir(parents=True, exist_ok=True)
    db = folder / "registry.sqlite"
    if db.exists():
        parser.error("请选择没有现有注册库的新目录")
    uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.create_experiment("serving-release", artifact_location=(folder / "artifacts").as_uri())
    client.create_registered_model(NAME)
    versions = {}
    for release, max_tokens in (("refund-01", 32), ("refund-02", 48)):
        manifest = dict(
            release_id=release, model=MODEL, revision=REVISION, tokenizer_revision=REVISION,
            template="tokenizer_config.json", adapter=None, quantization=None,
            max_tokens=max_tokens, fixture="refund-csv-v1",
        )
        text = json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n"
        sha = hashlib.sha256(text.encode()).hexdigest()
        with mlflow.start_run(experiment_id=experiment) as run:
            mlflow.log_params({"revision": REVISION, "max_tokens": max_tokens, "fixture": manifest["fixture"]})
            # 这里的评测是清单的结构验收。业务模型评测由独立报告提供。
            assert set(manifest) == {"release_id", "model", "revision", "tokenizer_revision", "template",
                                     "adapter", "quantization", "max_tokens", "fixture"}
            mlflow.log_metric("manifest_checks_passed", 1)
            mlflow.log_text(text, "manifest.json")
            version = client.create_model_version(
                NAME, run.info.artifact_uri, run_id=run.info.run_id,
                tags={"manifest_sha256": sha, "release_id": release, "gate": "manifest_ok"},
            )
        versions[version.version] = dict(release_id=release, manifest_sha256=sha,
                                         registered_version=version.version, has_run=bool(version.run_id))
    transitions = []
    for step, number in (("deploy", "1"), ("promote", "2"), ("rollback", "1")):
        client.set_registered_model_alias(NAME, "champion", number)
        target = client.get_model_version_by_alias(NAME, "champion")
        actual = Path(client.download_artifacts(target.run_id, "manifest.json")).read_bytes()
        if hashlib.sha256(actual).hexdigest() != target.tags["manifest_sha256"]:
            raise RuntimeError("发布清单内容变更")
        manifest = json.loads(actual)
        assert str(target.version) == number and manifest["release_id"] == target.tags["release_id"]
        transitions.append(dict(step=step, release_id=manifest["release_id"],
                                version=target.version, request_header={"X-Release-Id": manifest["release_id"]}))
    (folder / "results.json").write_text(json.dumps(
        dict(mlflow=mlflow.__version__, versions=versions, transitions=transitions),
        ensure_ascii=False, indent=2,
    ) + "\n")


if __name__ == "__main__":
    main()
