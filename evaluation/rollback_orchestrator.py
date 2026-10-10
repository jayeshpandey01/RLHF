"""Autonomous Checkpoint Rollback & Hot-Swap Orchestrator (Milestone 4, Phase 4.4).

Manages atomic production model state, golden anchor tracking, hot-swap rollbacks,
and release deployment execution based on statistical release gate verdicts.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger

from evaluation.statistical_gating import GateDecision, GateReport


@dataclass
class DeploymentState:
    """Production deployment pointer state and audit history."""

    active_model_id: str
    active_checkpoint_path: str
    golden_anchor_id: str
    golden_anchor_checkpoint_path: str
    last_action: str = "INITIALIZED"
    last_timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return asdict(self)


class RollbackOrchestrator:
    """Manages atomic deployment pointers, golden anchors, and automatic rollbacks."""

    STATE_FILENAME = "active_deployment.json"

    def __init__(self, metadata_dir: Union[str, Path]):
        """Initialize the Rollback Orchestrator.

        Args:
            metadata_dir: Directory where deployment state metadata and audit ledgers are kept.
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.metadata_dir / self.STATE_FILENAME
        self.state = self._load_or_init_state()

    def _load_or_init_state(self) -> DeploymentState:
        """Load persistent deployment state from disk or initialize empty state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return DeploymentState(
                    active_model_id=data.get("active_model_id", "baseline_v0"),
                    active_checkpoint_path=data.get("active_checkpoint_path", ""),
                    golden_anchor_id=data.get("golden_anchor_id", "baseline_v0"),
                    golden_anchor_checkpoint_path=data.get("golden_anchor_checkpoint_path", ""),
                    last_action=data.get("last_action", "LOADED"),
                    last_timestamp=data.get("last_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
                    history=data.get("history", []),
                )
            except Exception as e:
                logger.warning(f"Failed to read {self.state_file} ({e}). Creating new state.")

        init_state = DeploymentState(
            active_model_id="baseline_v0",
            active_checkpoint_path="",
            golden_anchor_id="baseline_v0",
            golden_anchor_checkpoint_path="",
        )
        self._save_state(init_state)
        return init_state

    def _save_state(self, state: DeploymentState) -> None:
        """Atomically persist state to disk via temporary file rename."""
        tmp_file = self.metadata_dir / f"{self.STATE_FILENAME}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
        os.replace(tmp_file, self.state_file)
        self.state = state

    def register_golden_anchor(self, model_id: str, checkpoint_path: str) -> None:
        """Register a verified, regression-free model as the golden anchor checkpoint."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.state.golden_anchor_id = model_id
        self.state.golden_anchor_checkpoint_path = str(checkpoint_path)
        if not self.state.active_checkpoint_path:
            self.state.active_model_id = model_id
            self.state.active_checkpoint_path = str(checkpoint_path)
        self.state.last_action = "REGISTER_ANCHOR"
        self.state.last_timestamp = now
        self.state.history.append({
            "action": "REGISTER_ANCHOR",
            "model_id": model_id,
            "checkpoint_path": str(checkpoint_path),
            "timestamp": now,
        })
        self._save_state(self.state)
        logger.info(f"Registered Golden Anchor: {model_id} -> {checkpoint_path}")

    def promote_candidate(
        self,
        candidate_id: str,
        checkpoint_path: str,
        report: Optional[GateReport] = None,
    ) -> dict[str, Any]:
        """Atomically promote a candidate model to active production status."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        prev_model = self.state.active_model_id
        prev_path = self.state.active_checkpoint_path

        self.state.active_model_id = candidate_id
        self.state.active_checkpoint_path = str(checkpoint_path)
        self.state.last_action = "PROMOTED"
        self.state.last_timestamp = now

        audit_entry = {
            "action": "PROMOTED",
            "candidate_id": candidate_id,
            "checkpoint_path": str(checkpoint_path),
            "previous_model": prev_model,
            "previous_path": prev_path,
            "gate_decision": report.decision.value if report else "MANUAL",
            "mean_delta": report.overall_bootstrap.mean_delta if report else 0.0,
            "timestamp": now,
        }
        self.state.history.append(audit_entry)
        self._save_state(self.state)
        logger.info(f"🚀 Model Promoted: {prev_model} -> {candidate_id} ({checkpoint_path})")
        return audit_entry

    def rollback_to_anchor(
        self,
        report: Optional[GateReport] = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Atomically restore active deployment pointer to the verified golden anchor."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        failed_model = self.state.active_model_id
        failed_path = self.state.active_checkpoint_path

        anchor_id = self.state.golden_anchor_id
        anchor_path = self.state.golden_anchor_checkpoint_path

        self.state.active_model_id = anchor_id
        self.state.active_checkpoint_path = anchor_path
        self.state.last_action = "ROLLED_BACK"
        self.state.last_timestamp = now

        audit_entry = {
            "action": "ROLLED_BACK",
            "failed_model": failed_model,
            "failed_path": failed_path,
            "restored_anchor": anchor_id,
            "anchor_path": anchor_path,
            "reason": reason or (report.recommendation_summary if report else "Gate trigger"),
            "gate_decision": report.decision.value if report else "TRIGGER_ROLLBACK",
            "timestamp": now,
        }
        self.state.history.append(audit_entry)
        self._save_state(self.state)
        logger.warning(
            f"⚠️ Emergency Rollback Executed: {failed_model} rolled back to golden anchor {anchor_id}"
        )
        return audit_entry

    def execute_gate_decision(
        self,
        report: GateReport,
        candidate_model_id: str,
        candidate_checkpoint_path: str,
    ) -> dict[str, Any]:
        """Execute automated deployment action driven by GateReport verdict.

        - PROCEED_DEPLOY: Promotes candidate to active status.
        - TRIGGER_ROLLBACK: Reverts to golden anchor immediately.
        - BLOCK_RELEASE / ALERT_DRIFT: Leaves active state untouched and logs audit record.
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if report.decision == GateDecision.PROCEED_DEPLOY:
            return self.promote_candidate(
                candidate_id=candidate_model_id,
                checkpoint_path=candidate_checkpoint_path,
                report=report,
            )
        elif report.decision == GateDecision.TRIGGER_ROLLBACK:
            # Only perform active rollback if the regressed model is the currently active deployment
            if candidate_model_id == self.state.active_model_id:
                return self.rollback_to_anchor(
                    report=report,
                    reason=report.recommendation_summary,
                )
            else:
                audit_entry = {
                    "action": "HELD_TRIGGER_ROLLBACK",
                    "candidate_id": candidate_model_id,
                    "current_active": self.state.active_model_id,
                    "gate_decision": report.decision.value,
                    "summary": (
                        f"Candidate {candidate_model_id} triggered rollback evaluation before release; "
                        f"candidate rejected, active model {self.state.active_model_id} retained untouched."
                    ),
                    "timestamp": now,
                }
                self.state.history.append(audit_entry)
                self._save_state(self.state)
                logger.warning(
                    f"Candidate {candidate_model_id} rejected due to regressions. Active deployment remains {self.state.active_model_id}."
                )
                return audit_entry
        else:
            # BLOCK_RELEASE or ALERT_DRIFT: Keep current model
            audit_entry = {
                "action": f"HELD_{report.decision.value}",
                "candidate_id": candidate_model_id,
                "current_active": self.state.active_model_id,
                "gate_decision": report.decision.value,
                "summary": report.recommendation_summary,
                "timestamp": now,
            }
            self.state.history.append(audit_entry)
            self._save_state(self.state)
            logger.info(
                f"Release Gate Decision: {report.decision.value}. Active deployment remains {self.state.active_model_id}."
            )
            return audit_entry

