"""Validation gates and their result types.

Every Finding carries a `hint` saying what to do about it.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

ERROR, WARN, INFO = "error", "warn", "info"


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    hint: str = ""

    def __str__(self):
        tag = {"error": "FAIL", "warn": "WARN", "info": "info"}[self.severity]
        s = f"  [{tag}] {self.code}: {self.message}"
        if self.hint:
            s += f"\n         -> {self.hint}"
        return s


@dataclass
class Result:
    name: str
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    ran: bool = True

    @property
    def errors(self):
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self):
        return [f for f in self.findings if f.severity == WARN]

    @property
    def passed(self) -> bool:
        return self.ran and not self.errors

    def add(self, severity, code, message, hint=""):
        self.findings.append(Finding(severity, code, message, hint))
        return self

    def error(self, code, message, hint=""):
        return self.add(ERROR, code, message, hint)

    def warn(self, code, message, hint=""):
        return self.add(WARN, code, message, hint)

    def info(self, code, message, hint=""):
        return self.add(INFO, code, message, hint)

    def report(self) -> str:
        head = f"{self.name}: {'PASS' if self.passed else 'FAIL'}"
        if not self.ran:
            head += " (did not run)"
        lines = [head] + [str(f) for f in self.findings]
        if self.metrics:
            lines.append("  metrics: " + ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in self.metrics.items()))
        return "\n".join(lines)

    def feedback(self) -> str:
        """The errors and warnings, formatted."""
        out = []
        for f in self.findings:
            if f.severity in (ERROR, WARN):
                out.append(f"- {f.code}: {f.message}"
                           + (f"\n  FIX: {f.hint}" if f.hint else ""))
        return "\n".join(out)


def summarize(results) -> str:
    return "\n\n".join(r.report() for r in results)
