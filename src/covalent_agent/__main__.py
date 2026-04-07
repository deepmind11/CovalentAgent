"""CLI entry point for CovalentAgent.

Usage:
    python -m covalent_agent --target KRAS --residue C12
    python -m covalent_agent -t EGFR -r C797 -i "non-small cell lung cancer" -o report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from covalent_agent import __version__
from covalent_agent.supervisor import CovalentAgentPipeline


def main() -> None:
    """Parse arguments and run the covalent drug design pipeline."""
    parser = argparse.ArgumentParser(
        prog="covalent-agent",
        description="AI-powered covalent drug design pipeline",
    )
    parser.add_argument(
        "--target",
        "-t",
        required=True,
        help="Protein name (e.g. KRAS, EGFR, BTK)",
    )
    parser.add_argument(
        "--residue",
        "-r",
        required=True,
        help="Target residue (e.g. C12, C797, C481)",
    )
    parser.add_argument(
        "--indication",
        "-i",
        default="",
        help="Disease indication (e.g. 'non-small cell lung cancer')",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path for JSON report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"CovalentAgent v{__version__}")
    print(f"Target: {args.target} {args.residue}")
    if args.indication:
        print(f"Indication: {args.indication}")
    print()
    print("Running pipeline...")
    print("-" * 60)

    try:
        pipeline = CovalentAgentPipeline()
        report = asyncio.run(
            pipeline.run(
                target=args.target,
                residue=args.residue,
                indication=args.indication,
            )
        )
    except RuntimeError as exc:
        print(f"\nPipeline error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # -- Print key results to stdout ------------------------------------------

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Ligandability
    print(f"\nLigandability Assessment:")
    print(f"  {report.ligandability_assessment[:200]}")

    # Top candidate's warhead class (from highest-scoring molecule, not the
    # WarheadSelector's #1 pick — which may differ if a lower-ranked warhead
    # produced a better-scoring candidate molecule).
    if report.ranked_candidates:
        top_warhead = report.ranked_candidates[0].warhead_class
        print(f"\nTop Candidate's Warhead Class: {top_warhead}")

    # Candidate counts
    print(f"\nCandidates Generated: {report.num_candidates_generated}")
    print(f"Candidates Passing:   {report.num_candidates_passing}")

    # Top 3 ranked candidates
    if report.ranked_candidates:
        print(f"\nTop {min(3, len(report.ranked_candidates))} Ranked Candidates:")
        print(f"  {'Rank':<6}{'Name':<45}{'Score':<10}{'Warhead':<20}")
        print(f"  {'-' * 6}{'-' * 45}{'-' * 10}{'-' * 20}")

        for rc in report.ranked_candidates[:3]:
            name_display = rc.name[:42] + "..." if len(rc.name) > 42 else rc.name
            print(
                f"  {rc.rank:<6}{name_display:<45}"
                f"{rc.composite_score:<10.3f}{rc.warhead_class:<20}"
            )
    else:
        print("\nNo candidates passed the scoring threshold.")

    # Citations count
    print(f"\nLiterature Citations: {len(report.citations)}")
    print(f"Report Generated At:  {report.generated_at}")

    # -- Write JSON output if requested ----------------------------------------

    if args.output:
        output_path = args.output
        try:
            report_data = json.loads(report.model_dump_json())
            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2)
            print(f"\nFull report written to: {output_path}")
        except OSError as exc:
            print(f"\nFailed to write output file: {exc}", file=sys.stderr)
            sys.exit(1)

    print()


if __name__ == "__main__":
    main()
