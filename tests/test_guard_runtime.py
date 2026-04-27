import unittest

from cua_guard.classifiers.base import DangerClassifier
from cua_guard.runtime.agent import ScriptedAgent
from cua_guard.runtime.environment import ToyComputerEnvironment
from cua_guard.runtime.guard import ConformalActionGuard
from cua_guard.runtime.runner import EscalationResolution, run_episode
from cua_guard.types import ActionProposal, LabeledAction


class DeleteScorer(DangerClassifier):
    def fit(self, records: list[LabeledAction]) -> "DeleteScorer":
        return self

    def score(self, proposal: ActionProposal) -> float:
        text = f"{proposal.action_type} {proposal.target} {proposal.text}".lower()
        return 0.95 if "delete" in text else 0.05

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError


class GuardRuntimeTest(unittest.TestCase):
    def test_guard_blocks_high_scoring_action_before_environment_executes_it(self):
        guard = ConformalActionGuard(DeleteScorer(), threshold=0.5, mode="block")
        agent = ScriptedAgent(
            [
                {"type": "click", "target": "Search docs"},
                {"type": "click", "target": "Delete project"},
            ]
        )
        env = ToyComputerEnvironment()

        result = run_episode(agent, env, guard, max_steps=5)

        self.assertEqual(len(result.steps), 2)
        self.assertTrue(result.steps[0].executed)
        self.assertFalse(result.steps[1].executed)
        self.assertTrue(result.stopped_by_guard)
        self.assertFalse(result.unsafe_event)

    def test_escalation_without_callback_stops_as_unresolved(self):
        guard = ConformalActionGuard(DeleteScorer(), threshold=0.5, mode="escalate")
        agent = ScriptedAgent([{"type": "click", "target": "Delete project"}])
        env = ToyComputerEnvironment()

        result = run_episode(agent, env, guard, max_steps=5)

        self.assertEqual(result.escalation_count, 1)
        self.assertTrue(result.unresolved_escalation)
        self.assertTrue(result.stopped_by_guard)
        self.assertFalse(result.steps[0].executed)
        self.assertFalse(result.unsafe_event)

    def test_escalation_callback_can_approve_execution(self):
        guard = ConformalActionGuard(DeleteScorer(), threshold=0.5, mode="escalate")
        agent = ScriptedAgent([{"type": "click", "target": "Delete project"}])
        env = ToyComputerEnvironment()
        seen = []

        def approve(decision):
            seen.append(decision)
            return EscalationResolution(approved=True, reason="test approval")

        result = run_episode(agent, env, guard, max_steps=1, on_escalate=approve)

        self.assertEqual(len(seen), 1)
        self.assertEqual(result.escalation_count, 1)
        self.assertFalse(result.unresolved_escalation)
        self.assertFalse(result.stopped_by_guard)
        self.assertTrue(result.steps[0].executed)
        self.assertTrue(result.unsafe_event)


if __name__ == "__main__":
    unittest.main()
