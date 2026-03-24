"""Claim Provenance Chain — causal verification graph.

Tracks which claims depend on which earlier claims. When a claim at step N
references a result from step M, a dependency edge is recorded. If step M's
claim is later invalidated (VERIFIED_FAIL), all downstream dependents are
automatically flagged as TAINTED.

This is the core research novelty of loop-guard: no other system tracks
causal dependencies between agent claims across loop iterations.

Usage:
    chain = ProvenanceChain()
    chain.record(step=0, claim=claim0, finding=finding0)
    chain.record(step=1, claim=claim1, finding=finding1, depends_on=[0])

    # If step 0 is invalidated:
    tainted = chain.invalidate(step=0)
    # Returns all steps that depended on step 0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loop_guard.models import Claim, ClaimType, Finding, Verdict, VerificationLevel


@dataclass
class ProvenanceNode:
    """A single node in the provenance graph."""

    step_id: int
    claim: Claim
    finding: Finding
    depends_on: list[int] = field(default_factory=list)  # step IDs this depends on
    depended_by: list[int] = field(default_factory=list)  # step IDs that depend on this
    tainted: bool = False
    tainted_reason: str | None = None


class ProvenanceChain:
    """Tracks causal dependencies between claims across agent loop steps.

    When an agent at step 45 says "Based on the 94% accuracy from step 23...",
    the provenance chain records that step 45 depends on step 23. If step 23
    is later found to be wrong, step 45 and all its descendants are flagged.
    """

    def __init__(self) -> None:
        self.nodes: dict[int, list[ProvenanceNode]] = {}  # step_id -> nodes
        self._all_nodes: list[ProvenanceNode] = []

    def record(
        self,
        step_id: int,
        claim: Claim,
        finding: Finding,
        depends_on: list[int] | None = None,
    ) -> ProvenanceNode:
        """Record a claim and its verification result in the provenance graph."""
        node = ProvenanceNode(
            step_id=step_id,
            claim=claim,
            finding=finding,
            depends_on=depends_on or [],
        )

        # Register forward edges
        for dep_step in node.depends_on:
            if dep_step in self.nodes:
                for dep_node in self.nodes[dep_step]:
                    dep_node.depended_by.append(step_id)

        if step_id not in self.nodes:
            self.nodes[step_id] = []
        self.nodes[step_id].append(node)
        self._all_nodes.append(node)

        # Check if any dependency is already tainted or failed
        for dep_step in node.depends_on:
            if dep_step in self.nodes:
                for dep_node in self.nodes[dep_step]:
                    if dep_node.tainted or dep_node.finding.verdict == Verdict.VERIFIED_FAIL:
                        node.tainted = True
                        node.tainted_reason = (
                            f"Depends on step {dep_step} which is "
                            f"{'tainted' if dep_node.tainted else 'VERIFIED_FAIL'}"
                        )
                        break

        return node

    def invalidate(self, step_id: int, reason: str | None = None) -> list[Finding]:
        """Invalidate a step and cascade taint to all dependents.

        Returns list of new findings for tainted downstream steps.
        """
        if step_id not in self.nodes:
            return []

        tainted_findings = []
        visited = set()

        def _cascade(sid: int, origin: int) -> None:
            if sid in visited:
                return
            visited.add(sid)

            if sid not in self.nodes:
                return

            for node in self.nodes[sid]:
                if not node.tainted and sid != origin:
                    node.tainted = True
                    node.tainted_reason = (
                        reason or f"Upstream dependency at step {origin} was invalidated"
                    )
                    tainted_findings.append(Finding(
                        step_id=sid,
                        claim=node.claim,
                        verdict=Verdict.FLAG_FOR_REVIEW,
                        level=VerificationLevel.RULE_BASED,
                        explanation=(
                            f"TAINTED: This claim depends on step {origin} which was "
                            f"invalidated. Original claim: \"{node.claim.text[:100]}\""
                        ),
                        expected=f"Valid upstream at step {origin}",
                        actual=f"Step {origin} invalidated: {reason or 'verification failed'}",
                    ))

                for dependent_step in node.depended_by:
                    _cascade(dependent_step, origin)

        # Mark the source step
        for node in self.nodes[step_id]:
            node.tainted = True
            node.tainted_reason = reason or "Directly invalidated"

        # Cascade to dependents
        for node in self.nodes[step_id]:
            for dependent_step in node.depended_by:
                _cascade(dependent_step, step_id)

        return tainted_findings

    def auto_detect_dependencies(
        self, step_id: int, raw_output: str
    ) -> list[int]:
        """Automatically detect references to previous steps in agent output.

        Looks for patterns like:
        - "from step 23"
        - "as shown in experiment 5"
        - "the 94% accuracy we achieved earlier"
        - "based on the previous result"
        - "step 23 showed..."
        """
        dependencies = []

        # Explicit step references: "step 23", "experiment 5", "iteration 12"
        for match in re.finditer(
            r"(?:step|experiment|iteration|trial|run)\s+#?(\d+)",
            raw_output,
            re.IGNORECASE,
        ):
            ref_step = int(match.group(1))
            if ref_step < step_id and ref_step in self.nodes:
                dependencies.append(ref_step)

        # Metric references that match previous claims
        # e.g., "the 94% accuracy" — find which step produced that metric
        for match in re.finditer(r"the\s+([\d.]+%?)\s+\w+", raw_output):
            value_str = match.group(1)
            for sid, nodes in self.nodes.items():
                if sid >= step_id:
                    continue
                for node in nodes:
                    if (
                        node.claim.claim_type == ClaimType.METRIC
                        and value_str in node.claim.text
                    ):
                        dependencies.append(sid)

        # "previous", "earlier", "above" — reference the immediate predecessor
        if re.search(r"\b(previous|earlier|above|prior)\b", raw_output, re.IGNORECASE):
            prev_steps = [s for s in self.nodes if s < step_id]
            if prev_steps:
                dependencies.append(max(prev_steps))

        return sorted(set(dependencies))

    def get_tainted_steps(self) -> list[int]:
        """Return all step IDs that are currently tainted."""
        return sorted({
            node.step_id
            for node in self._all_nodes
            if node.tainted
        })

    def get_dependency_chain(self, step_id: int) -> list[int]:
        """Get the full dependency chain for a step (recursive ancestors)."""
        chain = []
        visited = set()

        def _trace(sid: int) -> None:
            if sid in visited:
                return
            visited.add(sid)
            if sid in self.nodes:
                for node in self.nodes[sid]:
                    for dep in node.depends_on:
                        chain.append(dep)
                        _trace(dep)

        _trace(step_id)
        return sorted(set(chain))

    def summary(self) -> dict:
        """Return provenance chain statistics."""
        total_nodes = len(self._all_nodes)
        total_edges = sum(len(n.depends_on) for n in self._all_nodes)
        tainted = len(self.get_tainted_steps())
        max_depth = 0

        for node in self._all_nodes:
            depth = len(self.get_dependency_chain(node.step_id))
            max_depth = max(max_depth, depth)

        return {
            "total_claims_tracked": total_nodes,
            "total_dependency_edges": total_edges,
            "tainted_steps": tainted,
            "max_dependency_depth": max_depth,
            "steps_with_dependencies": sum(
                1 for n in self._all_nodes if n.depends_on
            ),
        }
