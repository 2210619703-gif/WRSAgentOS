"""Exercise script startup with temporary configuration, without online calls or nodes."""

import asyncio
import os
import runpy
import shutil
from pathlib import Path

import pytest

from wrs_agent.planner.providers.llm import LLMConfig, LLMError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["01_plan.py", "02_execute.py"])
def script(request, tmp_path, monkeypatch):
    # Preserve the examples/models layout, but never load the operator's real .env.
    project = tmp_path / "project"
    entry = project / "examples/models" / request.param
    entry.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "examples/models" / request.param, entry)
    monkeypatch.setattr(os, "environ", {
        key: value for key, value in os.environ.items()
        if not key.startswith("LLM_") and key != "PYTHON_DOTENV_DISABLED"
    })
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    (elsewhere / ".env").write_text("LLM_MODEL=wrong-working-directory\n", encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Startup configuration checks must not create clients or nodes")

    monkeypatch.setattr("wrs_agent.planner.providers.llm.LLMClient", forbidden)
    monkeypatch.setattr("wrs_agent.processes.LocalStack", forbidden)
    return entry, project / ".env"


def test_script_loads_root_env_before_main_and_preserves_explicit_env(script, monkeypatch):
    entry, env_file = script
    env_file.write_text(
        'LLM_MODEL="fixture-model" # quoted dotenv syntax\n'
        'LLM_BASE_URL=https://model.invalid/v1\n'
        'LLM_API_KEY=fixture-key\n'
        'LLM_EXTRA_BODY=\'{"thinking":{"type":"disabled"}}\'\n',
        encoding="utf-8-sig",
    )
    monkeypatch.setenv("LLM_API_KEY", "explicit-key")
    configured = []

    def inspect_startup(coroutine):
        coroutine.close()  # Inspect startup only; never execute the online example.
        configured.append(LLMConfig.from_env())

    monkeypatch.setattr(asyncio, "run", inspect_startup)
    runpy.run_path(str(entry), run_name="__main__")
    assert len(configured) == 1
    assert configured[0].model == "fixture-model"
    assert configured[0].base_url == "https://model.invalid/v1"
    assert configured[0].extra_body == {"thinking": {"type": "disabled"}}
    assert os.environ["LLM_API_KEY"] == "explicit-key"


def test_import_does_not_load_env_or_start_example(script, monkeypatch):
    entry, env_file = script
    env_file.write_text("LLM_MODEL=must-not-load\n", encoding="utf-8")
    monkeypatch.setattr(asyncio, "run", lambda _: pytest.fail("Import started the example"))
    runpy.run_path(str(entry))
    assert "LLM_MODEL" not in os.environ


def test_missing_root_env_does_not_load_cwd_env_or_choose_model(script, monkeypatch):
    entry, _ = script
    configured = []

    def inspect_startup(coroutine):
        coroutine.close()
        with pytest.raises(LLMError) as error:
            LLMConfig.from_env()
        configured.append(error.value.code)

    monkeypatch.setattr(asyncio, "run", inspect_startup)
    runpy.run_path(str(entry), run_name="__main__")
    assert configured == ["llm_model_missing"]
    assert "LLM_MODEL" not in os.environ
