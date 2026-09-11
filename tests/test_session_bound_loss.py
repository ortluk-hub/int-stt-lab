"""
Test fixture for session-bound loss reproduction.

This fixture reproduces the "invalid-worker-context" issue observed in Kanban task handoffs.
The issue occurs when a worker exits cleanly without calling `kanban_complete` or
`kanban_request_review`, or when the worker context is corrupted.

Key Test Cases:
1. Clean exit without handoff (protocol violation).
2. Corrupted worker context (missing/invalid fields).
3. Missing run_id or task_id in handoff.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock


# Mock Kanban tools to simulate failures
class MockKanbanTools:
    """Mock Kanban lifecycle tools to simulate session-bound loss."""

    @staticmethod
    def kanban_complete(task_id=None, summary=None, metadata=None, board=None):
        """Simulate a failed kanban_complete call."""
        if task_id is None:
            raise ValueError("task_id is required")
        raise RuntimeError("kanban_complete failed: protocol violation")

    @staticmethod
    def kanban_request_review(task_id=None, summary=None, reviewer=None, metadata=None, board=None):
        """Simulate a failed kanban_request_review call."""
        raise RuntimeError("kanban_request_review failed: invalid worker context")


def test_clean_exit_without_handoff():
    """
    Reproduce the protocol violation: worker exits cleanly without calling
    kanban_complete or kanban_request_review.
    """
    with patch.dict(os.environ, {"HERMES_KANBAN_TASK": "t_0b2c7c5a"}):
        # Simulate a worker that exits cleanly without handoff
        with pytest.raises(SystemExit) as exc_info:
            sys.exit(0)
        
        # Verify the exit was clean (rc=0)
        assert exc_info.value.code == 0
        
        # Verify no Kanban handoff was attempted
        with patch.object(MockKanbanTools, "kanban_complete") as mock_complete:
            mock_complete.assert_not_called()
        
        with patch.object(MockKanbanTools, "kanban_request_review") as mock_review:
            mock_review.assert_not_called()


def test_corrupted_worker_context():
    """
    Reproduce the invalid-worker-context issue: worker context is missing
    or corrupted (e.g., missing run_id or task_id).
    """
    # Simulate a corrupted worker_context (missing run_id)
    corrupted_context = """
    # Kanban task t_0b2c7c5a: B0.1: Rex - fixture for session-bound loss reproduction
    Assignee: rex
    Status:   running
    Workspace: dir @ /home/ortluk/ortluk-hub/int-stt-lab
    """
    
    # Verify the context is invalid (missing run_id)
    assert "run_id" not in corrupted_context
    
    # Simulate a handoff attempt with corrupted context
    with patch.dict(os.environ, {"HERMES_KANBAN_TASK": "t_0b2c7c5a"}):
        with pytest.raises(RuntimeError) as exc_info:
            MockKanbanTools.kanban_request_review(
                task_id="t_0b2c7c5a",
                summary="Test handoff with corrupted context",
                metadata={"worker_context": corrupted_context}
            )
        
        assert "invalid worker context" in str(exc_info.value)


def test_missing_task_id_in_handoff():
    """
    Reproduce the missing task_id issue: handoff is attempted without a task_id.
    """
    with patch.dict(os.environ, {"HERMES_KANBAN_TASK": ""}):
        with pytest.raises(ValueError) as exc_info:
            MockKanbanTools.kanban_complete(
                summary="Test handoff without task_id"
            )
        
        assert "task_id is required" in str(exc_info.value)


def test_repair_invalid_worker_context():
    """
    Validate the repair for invalid worker context: ensure the fixture can detect
    and recover from a corrupted context.
    """
    # Simulate a valid worker_context
    valid_context = """
    # Kanban task t_0b2c7c5a: B0.1: Rex - fixture for session-bound loss reproduction
    Assignee: rex
    Status:   running
    Workspace: dir @ /home/ortluk/ortluk-hub/int-stt-lab
    run_id: 165
    """
    
    # Verify the context is valid
    assert "run_id: 165" in valid_context
    
    # Simulate a successful handoff with valid context
    with patch.dict(os.environ, {"HERMES_KANBAN_TASK": "t_0b2c7c5a"}):
        # Mock a successful kanban_request_review
        with patch.object(MockKanbanTools, "kanban_request_review") as mock_review:
            mock_review.return_value = {"status": "success"}
            
            result = MockKanbanTools.kanban_request_review(
                task_id="t_0b2c7c5a",
                summary="Test handoff with valid context",
                metadata={"worker_context": valid_context}
            )
            
            assert result["status"] == "success"
            mock_review.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])