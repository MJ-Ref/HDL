#!/usr/bin/env python3
"""
Documentation health checker for LPCA/HDL repository.

Validates that documentation files are consistent, up-to-date, and properly cross-linked.
Designed to run as a pre-commit hook or CI check.

Usage:
    python scripts/check_docs.py          # Check all docs
    python scripts/check_docs.py --fix    # Show what needs fixing (no auto-fix)
    python scripts/check_docs.py --verbose # Show detailed output
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


class CheckResult(NamedTuple):
    """Result of a single check."""

    name: str
    passed: bool
    message: str
    severity: str  # 'error', 'warning', 'info'


class DocsChecker:
    """Check documentation health."""

    def __init__(self, repo_root: Path, verbose: bool = False):
        self.root = repo_root
        self.verbose = verbose
        self.results: list[CheckResult] = []

    def log(self, msg: str) -> None:
        """Print if verbose mode."""
        if self.verbose:
            print(f"  {msg}")

    def add_result(
        self, name: str, passed: bool, message: str, severity: str = "error"
    ) -> None:
        """Add a check result."""
        self.results.append(CheckResult(name, passed, message, severity))

    def read_file(self, path: Path) -> str | None:
        """Read file contents, return None if not found."""
        full_path = self.root / path
        if not full_path.exists():
            return None
        return full_path.read_text()

    def extract_status(self, content: str) -> str | None:
        """Extract status line from markdown content."""
        # Look for **Status:** or **Current Phase:** patterns
        patterns = [
            r"\*\*Status:\*\*\s*(.+?)(?:\n|$)",
            r"\*\*Current Phase:\*\*\s*(.+?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
        return None

    def extract_last_updated(self, content: str) -> datetime | None:
        """Extract 'Last Updated' date from markdown content."""
        # Look for **Last Updated:** or Last Updated: patterns
        match = re.search(
            r"(?:\*\*)?Last Updated:?(?:\*\*)?\s*(\w+\s+\d{1,2},?\s+\d{4})", content
        )
        if match:
            date_str = match.group(1)
            # Try parsing various formats
            for fmt in ["%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
        return None

    def extract_gate_results(self, content: str) -> dict[str, float] | None:
        """Extract Gate 1 results (Normal, Null, Shuffle percentages)."""
        results = {}
        # Look for patterns like "Normal | 22.0%" or "Normal: 22.0%"
        for condition in ["Normal", "Null", "Shuffle", "Random"]:
            match = re.search(rf"{condition}\s*[|\:]\s*\*?\*?(\d+\.?\d*)%", content)
            if match:
                results[condition.lower()] = float(match.group(1))
        return results if results else None

    def check_status_consistency(self) -> None:
        """Check that PLAN.md and PROJECT_STATUS.md have consistent status."""
        self.log("Checking status consistency...")

        plan = self.read_file(Path("PLAN.md"))
        status = self.read_file(Path("PROJECT_STATUS.md"))

        if not plan:
            self.add_result("status_consistency", False, "PLAN.md not found")
            return
        if not status:
            self.add_result("status_consistency", False, "PROJECT_STATUS.md not found")
            return

        plan_status = self.extract_status(plan)
        status_status = self.extract_status(status)

        self.log(f"PLAN.md status: {plan_status}")
        self.log(f"PROJECT_STATUS.md status: {status_status}")

        if not plan_status:
            self.add_result(
                "status_consistency", False, "Could not extract status from PLAN.md"
            )
            return
        if not status_status:
            self.add_result(
                "status_consistency",
                False,
                "Could not extract status from PROJECT_STATUS.md",
            )
            return

        # Extract key status indicators
        def extract_key_status(s: str) -> str | None:
            """Extract the key status: FAILED, PASSED, INVALID, COMPLETE, etc."""
            s_upper = s.upper()
            for status in [
                "FAILED",
                "INVALID",
                "PASSED",
                "COMPLETE",
                "IN PROGRESS",
                "BLOCKED",
            ]:
                if status in s_upper:
                    return status
            return None

        plan_key = extract_key_status(plan_status)
        status_key = extract_key_status(status_status)

        self.log(f"Key status - PLAN: {plan_key}, STATUS: {status_key}")

        if plan_key and status_key and plan_key == status_key:
            self.add_result(
                "status_consistency", True, f"Status is consistent: {plan_key}"
            )
        elif not plan_key or not status_key:
            self.add_result(
                "status_consistency",
                False,
                f"Could not extract key status:\n  PLAN.md: {plan_status}\n  PROJECT_STATUS.md: {status_status}",
                "warning",
            )
        else:
            self.add_result(
                "status_consistency",
                False,
                f"Status mismatch:\n  PLAN.md: {plan_key} ({plan_status})\n  PROJECT_STATUS.md: {status_key} ({status_status})",
            )

    def check_metrics_consistency(self) -> None:
        """Check that Gate 1 metrics are consistent across docs."""
        self.log("Checking metrics consistency...")

        plan = self.read_file(Path("PLAN.md"))
        status = self.read_file(Path("PROJECT_STATUS.md"))

        if not plan or not status:
            self.add_result(
                "metrics_consistency", False, "Missing required files", "warning"
            )
            return

        plan_results = self.extract_gate_results(plan)
        status_results = self.extract_gate_results(status)

        self.log(f"PLAN.md results: {plan_results}")
        self.log(f"PROJECT_STATUS.md results: {status_results}")

        if not plan_results:
            self.add_result(
                "metrics_consistency", True, "No Gate results in PLAN.md yet", "info"
            )
            return
        if not status_results:
            self.add_result(
                "metrics_consistency",
                False,
                "Gate results in PLAN.md but not PROJECT_STATUS.md",
            )
            return

        # Compare key metrics
        mismatches = []
        for key in ["normal", "null", "shuffle"]:
            if key in plan_results and key in status_results:
                if abs(plan_results[key] - status_results[key]) > 0.1:
                    mismatches.append(
                        f"{key}: PLAN={plan_results[key]}% vs STATUS={status_results[key]}%"
                    )

        if mismatches:
            self.add_result(
                "metrics_consistency",
                False,
                f'Metrics mismatch: {", ".join(mismatches)}',
            )
        else:
            self.add_result("metrics_consistency", True, "Gate metrics are consistent")

    def check_freshness(self) -> None:
        """Check that evolving docs aren't stale."""
        self.log("Checking document freshness...")

        # Docs that should be updated regularly during active work
        evolving_docs = [
            ("PLAN.md", 14),  # Should be updated within 14 days during active work
            ("PROJECT_STATUS.md", 14),
            ("SESSION_HANDOFF.md", 7),  # Should be updated more frequently
        ]

        today = datetime.now()

        for doc_path, max_days in evolving_docs:
            content = self.read_file(Path(doc_path))
            if not content:
                self.add_result(
                    f"freshness_{doc_path}", False, f"{doc_path} not found", "warning"
                )
                continue

            last_updated = self.extract_last_updated(content)
            if not last_updated:
                self.add_result(
                    f"freshness_{doc_path}",
                    False,
                    f'{doc_path}: Could not find "Last Updated" date',
                    "warning",
                )
                continue

            age_days = (today - last_updated).days
            self.log(f"{doc_path}: last updated {age_days} days ago")

            if age_days > max_days:
                self.add_result(
                    f"freshness_{doc_path}",
                    False,
                    f"{doc_path}: Last updated {age_days} days ago (threshold: {max_days} days)",
                    "warning",
                )
            else:
                self.add_result(
                    f"freshness_{doc_path}",
                    True,
                    f"{doc_path}: Updated {age_days} days ago",
                )

    def check_experiments_coverage(self) -> None:
        """Check that docs/experiments/ has entries for Gate attempts."""
        self.log("Checking experiments coverage...")

        experiments_dir = self.root / "docs" / "experiments"
        if not experiments_dir.exists():
            self.add_result(
                "experiments_coverage", False, "docs/experiments/ directory not found"
            )
            return

        # Check for Gate 1 attempt files
        attempt_files = list(experiments_dir.glob("gate1-attempt-*.md"))

        if not attempt_files:
            self.add_result(
                "experiments_coverage",
                False,
                "No Gate 1 attempt files found in docs/experiments/",
            )
            return

        # Check that README.md index exists
        readme = experiments_dir / "README.md"
        if not readme.exists():
            self.add_result(
                "experiments_coverage",
                False,
                "docs/experiments/README.md index not found",
            )
            return

        # Check that README references all attempt files
        readme_content = readme.read_text()
        missing_refs = []
        for attempt_file in attempt_files:
            if attempt_file.name not in readme_content:
                missing_refs.append(attempt_file.name)

        if missing_refs:
            self.add_result(
                "experiments_coverage",
                False,
                f'Attempt files not referenced in README.md: {", ".join(missing_refs)}',
            )
        else:
            self.add_result(
                "experiments_coverage",
                True,
                f"Found {len(attempt_files)} Gate 1 attempt(s), all indexed",
            )

    def check_internal_links(self) -> None:
        """Check that internal markdown links are valid."""
        self.log("Checking internal links...")

        # Key docs to check
        docs_to_check = [
            "README.md",
            "PLAN.md",
            "PROJECT_STATUS.md",
            "docs/experiments/README.md",
        ]

        broken_links = []

        for doc_path in docs_to_check:
            content = self.read_file(Path(doc_path))
            if not content:
                continue

            # Find markdown links: [text](path)
            links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)

            for text, link_path in links:
                # Skip external links and anchors
                if link_path.startswith(("http://", "https://", "#", "mailto:")):
                    continue

                # Remove anchor from path
                clean_path = link_path.split("#")[0]
                if not clean_path:
                    continue

                # Resolve relative to doc location
                doc_dir = (self.root / doc_path).parent
                target = (doc_dir / clean_path).resolve()

                if not target.exists():
                    broken_links.append(f"{doc_path}: [{text}]({link_path})")

        if broken_links:
            self.add_result(
                "internal_links",
                False,
                "Broken links:\n  " + "\n  ".join(broken_links[:5]),  # Show first 5
            )
        else:
            self.add_result("internal_links", True, "All internal links valid")

    def run_all_checks(self) -> bool:
        """Run all checks and return True if all pass."""
        print("Running documentation health checks...\n")

        self.check_status_consistency()
        self.check_metrics_consistency()
        self.check_freshness()
        self.check_experiments_coverage()
        self.check_internal_links()

        # Print results
        errors = []
        warnings = []
        passed = []

        for result in self.results:
            if result.passed:
                passed.append(result)
            elif result.severity == "error":
                errors.append(result)
            else:
                warnings.append(result)

        # Print passed checks
        if passed:
            print("PASSED:")
            for r in passed:
                print(f"  ✓ {r.name}: {r.message}")
            print()

        # Print warnings
        if warnings:
            print("WARNINGS:")
            for r in warnings:
                print(f"  ⚠ {r.name}: {r.message}")
            print()

        # Print errors
        if errors:
            print("ERRORS:")
            for r in errors:
                print(f"  ✗ {r.name}: {r.message}")
            print()

        # Summary
        total = len(self.results)
        print(
            f"Summary: {len(passed)}/{total} passed, {len(warnings)} warnings, {len(errors)} errors"
        )

        return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Check documentation health")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fix", action="store_true", help="Show what needs fixing")
    args = parser.parse_args()

    # Find repo root (look for PLAN.md)
    current = Path.cwd()
    while current != current.parent:
        if (current / "PLAN.md").exists():
            break
        current = current.parent
    else:
        print("Error: Could not find repo root (no PLAN.md found)")
        sys.exit(1)

    checker = DocsChecker(current, verbose=args.verbose)
    success = checker.run_all_checks()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
