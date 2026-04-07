"""Tests for the covalent_agent CLI entry point.

Covers argparse parsing, --version output, the success path (with a mocked
CovalentAgentPipeline), error handling for RuntimeError / KeyboardInterrupt /
unexpected exceptions, and JSON output writing (success + OSError).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covalent_agent import __main__ as cli
from covalent_agent.schemas import Citation, FinalReport, RankedCandidate


# ---------------------------------------------------------------------------
# Helper: build a minimal FinalReport for tests
# ---------------------------------------------------------------------------


def _make_report(num_ranked: int = 2) -> FinalReport:
    ranked = [
        RankedCandidate(
            rank=i + 1,
            smiles=f"C{i}",
            name=f"candidate-{i}",
            composite_score=0.9 - 0.1 * i,
            warhead_class="Acrylamide",
            drug_likeness=0.7,
            qed_score=0.6,
            admet_summary="Good ADMET",
            synthetic_accessibility=2.5,
            literature_support="Supported by sotorasib paper",
            rationale="Strong candidate",
        )
        for i in range(num_ranked)
    ]
    return FinalReport(
        target_protein="KRAS",
        target_residue="C12",
        indication="NSCLC",
        ligandability_assessment="Highly ligandable cysteine in switch II.",
        num_candidates_generated=num_ranked,
        num_candidates_passing=num_ranked,
        ranked_candidates=ranked,
        methodology_summary="KRAS C12 covalent inhibitor design pipeline run.",
        citations=[
            Citation(
                title="Sotorasib paper",
                authors=["Smith J"],
                journal="Nature",
                year=2024,
                pmid="31645765",
            )
        ],
    )


@pytest.fixture
def mock_pipeline_success():
    """Patch CovalentAgentPipeline to return a fake report."""
    fake_report = _make_report()
    mock_pipeline = MagicMock()
    mock_pipeline.run = AsyncMock(return_value=fake_report)
    with patch(
        "covalent_agent.__main__.CovalentAgentPipeline",
        return_value=mock_pipeline,
    ) as patched:
        yield patched, fake_report


# ---------------------------------------------------------------------------
# argparse: required argument enforcement
# ---------------------------------------------------------------------------


class TestArgParseRequired:
    def test_missing_target_exits(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["covalent-agent", "--residue", "C12"])
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 2

    def test_missing_residue_exits(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["covalent-agent", "--target", "KRAS"])
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# --version flag
# ---------------------------------------------------------------------------


class TestVersionFlag:
    def test_version_flag_prints_and_exits(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["covalent-agent", "--version"])
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        # argparse uses exit code 0 for --version
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        # version goes to stdout for argparse "version" action
        assert "covalent-agent" in captured.out


# ---------------------------------------------------------------------------
# Success path: mocked pipeline
# ---------------------------------------------------------------------------


class TestSuccessPath:
    def test_basic_success(self, monkeypatch, capsys, mock_pipeline_success):
        patched, fake_report = mock_pipeline_success
        monkeypatch.setattr(
            "sys.argv",
            ["covalent-agent", "--target", "KRAS", "--residue", "C12"],
        )
        cli.main()
        captured = capsys.readouterr()
        assert "KRAS" in captured.out
        assert "C12" in captured.out
        assert "RESULTS" in captured.out
        # Top warhead label is the corrected version
        assert "Top Candidate's Warhead Class" in captured.out
        assert "Acrylamide" in captured.out
        # Pipeline was actually invoked
        patched.return_value.run.assert_awaited_once()

    def test_indication_printed_when_provided(
        self, monkeypatch, capsys, mock_pipeline_success
    ):
        monkeypatch.setattr(
            "sys.argv",
            [
                "covalent-agent",
                "--target",
                "KRAS",
                "--residue",
                "C12",
                "--indication",
                "non-small cell lung cancer",
            ],
        )
        cli.main()
        captured = capsys.readouterr()
        assert "Indication: non-small cell lung cancer" in captured.out

    def test_no_candidates_message(self, monkeypatch, capsys):
        empty_report = _make_report(num_ranked=0)
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=empty_report)
        with patch(
            "covalent_agent.__main__.CovalentAgentPipeline",
            return_value=mock_pipeline,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["covalent-agent", "--target", "KRAS", "--residue", "C12"],
            )
            cli.main()
        captured = capsys.readouterr()
        assert "No candidates passed the scoring threshold" in captured.out


# ---------------------------------------------------------------------------
# JSON output writing
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_writes_json_report(
        self,
        monkeypatch,
        capsys,
        tmp_path: Path,
        mock_pipeline_success,
    ):
        out_file = tmp_path / "report.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "covalent-agent",
                "--target",
                "KRAS",
                "--residue",
                "C12",
                "--output",
                str(out_file),
            ],
        )
        cli.main()
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["target_protein"] == "KRAS"
        assert data["target_residue"] == "C12"
        assert len(data["ranked_candidates"]) == 2
        captured = capsys.readouterr()
        assert "Full report written to" in captured.out

    def test_oserror_on_write_exits_with_one(
        self,
        monkeypatch,
        capsys,
        mock_pipeline_success,
    ):
        # Write to a path that cannot exist (file inside a non-dir)
        monkeypatch.setattr(
            "sys.argv",
            [
                "covalent-agent",
                "--target",
                "KRAS",
                "--residue",
                "C12",
                "--output",
                "/nonexistent_dir_xyz/report.json",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Failed to write output file" in captured.err


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_runtime_error_exits_one(self, monkeypatch, capsys):
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(side_effect=RuntimeError("pipeline boom"))
        with patch(
            "covalent_agent.__main__.CovalentAgentPipeline",
            return_value=mock_pipeline,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["covalent-agent", "--target", "KRAS", "--residue", "C12"],
            )
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Pipeline error" in captured.err
        assert "pipeline boom" in captured.err

    def test_keyboard_interrupt_exits_130(self, monkeypatch, capsys):
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(side_effect=KeyboardInterrupt())
        with patch(
            "covalent_agent.__main__.CovalentAgentPipeline",
            return_value=mock_pipeline,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["covalent-agent", "--target", "KRAS", "--residue", "C12"],
            )
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
        assert exc_info.value.code == 130
        captured = capsys.readouterr()
        assert "Interrupted by user" in captured.err

    def test_unexpected_exception_exits_one(self, monkeypatch, capsys):
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(side_effect=ValueError("weird"))
        with patch(
            "covalent_agent.__main__.CovalentAgentPipeline",
            return_value=mock_pipeline,
        ):
            monkeypatch.setattr(
                "sys.argv",
                ["covalent-agent", "--target", "KRAS", "--residue", "C12"],
            )
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Unexpected error" in captured.err

    def test_verbose_unexpected_exception_prints_traceback(
        self, monkeypatch, capsys
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(side_effect=ValueError("weird"))
        with patch(
            "covalent_agent.__main__.CovalentAgentPipeline",
            return_value=mock_pipeline,
        ):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "covalent-agent",
                    "--target",
                    "KRAS",
                    "--residue",
                    "C12",
                    "--verbose",
                ],
            )
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        # Traceback header from traceback.print_exc()
        assert "Traceback" in captured.err
