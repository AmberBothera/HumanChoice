import unittest
from pathlib import Path

from humanchoice.features import CORE_FEATURES

ROOT = Path(__file__).resolve().parents[1]


class ProjectTests(unittest.TestCase):
    def test_required_source_and_documentation_exist(self):
        required = [
            "README.md", "METHODOLOGY.md", "COMPARISON.md", "RESULTS.md",
            "scripts/run_analysis.py", "scripts/render_results.py",
            "app/streamlit_app.py", "data/README.md",
        ]
        for relative in required:
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_predictors_exclude_leakage_columns(self):
        forbidden = {"choice", "rt", "participant_id", "paper", "objective_score", "is_optimal", "optimality_gap"}
        predictors = set(CORE_FEATURES)
        self.assertTrue(forbidden.isdisjoint(predictors))
