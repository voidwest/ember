from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from sarf.cli import main


class BackendCliTests(unittest.TestCase):
    def run_cli(self, argv: list[str]) -> str:
        buffer = io.StringIO()
        with patch.dict("os.environ", {"PATH": ""}, clear=True), redirect_stdout(buffer):
            exit_code = main(argv)
        self.assertEqual(exit_code, 0)
        return buffer.getvalue()

    def test_backends_list_reports_optional_missing_backends(self) -> None:
        output = self.run_cli(["backends", "list"])

        self.assertIn("Sarf backend availability", output)
        self.assertIn("llama.cpp: no", output)
        self.assertIn("Ember: no", output)
        self.assertIn("missing binaries: tokenize, cli, simple", output)
        self.assertIn("missing binaries: ember", output)
        self.assertIn("Default llama.cpp does not emit Sarf-compatible hidden-state artifacts.", output)
        self.assertIn("Hidden-state extraction requires an emitting backend", output)

    def test_llama_cpp_doctor_reports_hidden_states_unavailable(self) -> None:
        output = self.run_cli(["backend", "llama-cpp", "doctor"])

        self.assertIn("llama.cpp", output)
        self.assertIn("available: no", output)
        self.assertIn("tokenization: no", output)
        self.assertIn("logits: no", output)
        self.assertIn("hidden_states: no", output)

    def test_ember_doctor_reports_validate_run_unchecked_when_missing(self) -> None:
        output = self.run_cli(["backend", "ember", "doctor"])

        self.assertIn("Ember", output)
        self.assertIn("available: no", output)
        self.assertIn("binary: missing", output)
        self.assertIn("validate-run help: not callable (not checked)", output)
        self.assertIn("validate_run: no", output)


if __name__ == "__main__":
    unittest.main()
