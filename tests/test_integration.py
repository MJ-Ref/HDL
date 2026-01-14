"""Integration tests for LPCA framework."""

import pytest
from lpca.envs.split_synthetic import SplitSyntheticEnv
from lpca.channels.text import NoCommChannel, FullTextChannel, BudgetedTextChannel
from lpca.core.metrics import MetricsCalculator


class MockAgent:
    """Simple mock agent for integration testing."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.turn_count = 0

    def reset(self):
        self.turn_count = 0

    def respond(self, observation: str, received_message: str = None):
        """Generate mock response."""
        self.turn_count += 1

        if self.turn_count == 1:
            # First turn: share information
            return {
                'text': f"Agent {self.agent_id} processing...",
                'message': f"Info from {self.agent_id}: {observation[:50]}",
                'final_answer': None,
            }
        elif self.turn_count >= 2 and received_message:
            # Second turn: provide answer
            return {
                'text': "Submitting answer.",
                'message': None,
                'final_answer': '{"x1": 0, "x2": 1, "x3": 0}',
            }
        else:
            return {
                'text': "Waiting for partner.",
                'message': "Need more info.",
                'final_answer': None,
            }


class TestFullPipeline:
    """Integration test for complete experimental pipeline."""

    def test_single_episode_p0(self):
        """Test single episode with no communication."""
        env = SplitSyntheticEnv(difficulty="easy")
        env.select_environment("constraint_satisfaction")
        channel = NoCommChannel()

        task = env.reset(seed=42)
        agent_A = MockAgent("A")
        agent_B = MockAgent("B")

        # Run episode
        channel.reset()
        agent_A.reset()
        agent_B.reset()

        current_message = None
        final_answer = None

        for turn in range(10):
            # Agent A
            response = agent_A.respond(task.obs_A, current_message)
            if response['final_answer']:
                final_answer = response['final_answer']
                break
            if response['message']:
                msg = channel.send(response['message'], 'A', 'B', turn)
                current_message = channel.receive(msg)
            else:
                current_message = None

            # Agent B
            response = agent_B.respond(task.obs_B, current_message)
            if response['final_answer']:
                final_answer = response['final_answer']
                break
            if response['message']:
                msg = channel.send(response['message'], 'B', 'A', turn)
                current_message = channel.receive(msg)
            else:
                current_message = None

        # Verify stats
        stats = channel.get_stats()
        assert stats['total_bits'] == 0  # No communication in P0
        # Note: message_count tracks send attempts, but 0 bits transmitted

    def test_single_episode_p1(self):
        """Test single episode with full text communication."""
        env = SplitSyntheticEnv(difficulty="easy")
        env.select_environment("constraint_satisfaction")
        channel = FullTextChannel()

        task = env.reset(seed=42)
        agent_A = MockAgent("A")
        agent_B = MockAgent("B")

        channel.reset()
        agent_A.reset()
        agent_B.reset()

        current_message = None
        final_answer = None

        for turn in range(10):
            response = agent_A.respond(task.obs_A, current_message)
            if response['final_answer']:
                final_answer = response['final_answer']
                break
            if response['message']:
                msg = channel.send(response['message'], 'A', 'B', turn)
                current_message = channel.receive(msg)
            else:
                current_message = None

            response = agent_B.respond(task.obs_B, current_message)
            if response['final_answer']:
                final_answer = response['final_answer']
                break
            if response['message']:
                msg = channel.send(response['message'], 'B', 'A', turn)
                current_message = channel.receive(msg)
            else:
                current_message = None

        # Verify communication happened
        stats = channel.get_stats()
        assert stats['total_bits'] > 0
        assert stats['message_count'] >= 1
        assert final_answer is not None

    def test_protocol_comparison(self):
        """Test comparing multiple protocols on same task."""
        env = SplitSyntheticEnv(difficulty="easy")
        env.select_environment("constraint_satisfaction")
        metrics = MetricsCalculator()

        results = {}
        for protocol_name, ChannelClass in [
            ("P0", NoCommChannel),
            ("P1", FullTextChannel),
            ("P2", BudgetedTextChannel),
        ]:
            channel = ChannelClass()
            bits_used = []

            # Run 5 episodes
            for ep in range(5):
                task = env.reset(seed=100 + ep)
                channel.reset()

                agent_A = MockAgent("A")
                agent_B = MockAgent("B")
                current_message = None

                for turn in range(10):
                    response = agent_A.respond(task.obs_A, current_message)
                    if response['final_answer']:
                        break
                    if response['message']:
                        msg = channel.send(response['message'], 'A', 'B', turn)
                        current_message = channel.receive(msg)
                    else:
                        current_message = None

                    response = agent_B.respond(task.obs_B, current_message)
                    if response['final_answer']:
                        break
                    if response['message']:
                        msg = channel.send(response['message'], 'B', 'A', turn)
                        current_message = channel.receive(msg)
                    else:
                        current_message = None

                bits_used.append(channel.get_stats()['total_bits'])

            results[protocol_name] = {
                'mean_bits': sum(bits_used) / len(bits_used),
            }

        # P0 should use no bits
        assert results['P0']['mean_bits'] == 0

        # P1 and P2 should use bits
        assert results['P1']['mean_bits'] > 0
        assert results['P2']['mean_bits'] > 0


class TestDeterminism:
    """Test reproducibility of experiments."""

    def test_same_seed_same_task(self):
        """Verify same seed produces identical tasks."""
        env1 = SplitSyntheticEnv(difficulty="medium")
        env2 = SplitSyntheticEnv(difficulty="medium")

        env1.select_environment("arithmetic")
        env2.select_environment("arithmetic")

        task1 = env1.reset(seed=999)
        task2 = env2.reset(seed=999)

        assert task1.obs_A == task2.obs_A
        assert task1.obs_B == task2.obs_B
        assert task1.ground_truth == task2.ground_truth

    def test_different_seed_different_task(self):
        """Verify different seeds produce different tasks."""
        env = SplitSyntheticEnv(difficulty="medium")
        env.select_environment("arithmetic")

        task1 = env.reset(seed=1)
        task2 = env.reset(seed=2)

        assert task1.ground_truth != task2.ground_truth


class TestMetricsIntegration:
    """Test metrics calculation with real episode data."""

    def test_success_rate_calculation(self):
        """Test calculating success rate from episodes."""
        calc = MetricsCalculator()

        outcomes = [True, True, False, True, False, True]
        rate = calc.success_rate(outcomes)

        assert abs(rate - 4/6) < 0.01

        lower, upper = calc.wilson_ci(4, 6)
        assert lower < rate < upper

    def test_turns_to_success_with_failures(self):
        """Test turns to success handles failures correctly."""
        calc = MetricsCalculator()

        turns = [3, 5, 10, 2, 8]
        outcomes = [True, False, True, True, False]

        median, iqr = calc.turns_to_success(turns, outcomes)

        assert median is not None
        # Only successful episodes: turns 3, 10, 2 -> median = 3
        assert median == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
