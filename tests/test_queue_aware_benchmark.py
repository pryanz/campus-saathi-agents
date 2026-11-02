"""
50-Seed Benchmark Test for Issue #22:
Verifies that Queue-Aware Planners achieve lower total completion time
than Queue-Blind Planners across 50 benchmark test seeds.
"""
import sys
import unittest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")

from envi.simulator import CampusSimulator
from algorithms.astar import AStarAgent, QueueAwareAStarAgent


class TestQueueAwareBenchmark(unittest.TestCase):
    def test_queue_aware_beats_blind_across_50_seeds(self):
        blind_costs = []
        qa_costs = []
        seeds_tested = 50

        print("\n" + "=" * 70)
        print("RUNNING 50-SEED BENCHMARK: QUEUE-AWARE VS QUEUE-BLIND PLANNERS")
        print("=" * 70)

        for seed in range(1, seeds_tested + 1):
            # 1. Queue-Blind A* Planner (Level 0)
            sim_blind = CampusSimulator(seed=seed, observation_level=0)
            agent_blind = AStarAgent(sim_blind.problem)
            res_blind = sim_blind.run_simulation(agent_blind, congestion=True, observation_level=0)
            cost_blind = res_blind["metrics"]["total_cost"]
            blind_costs.append(cost_blind)

            # 2. Queue-Aware A* Planner (Level 2)
            sim_qa = CampusSimulator(seed=seed, observation_level=2)
            agent_qa = QueueAwareAStarAgent(sim_qa.problem, observation_level=2)
            res_qa = sim_qa.run_simulation(agent_qa, congestion=True, observation_level=2)
            cost_qa = res_qa["metrics"]["total_cost"]
            qa_costs.append(cost_qa)

            diff = cost_blind - cost_qa
            if seed % 10 == 0 or seed == 1:
                print(f"  Seed {seed:2d}/50: Blind={cost_blind:.1f}m | QA={cost_qa:.1f}m | Saved={diff:+.1f}m")

        mean_blind = sum(blind_costs) / seeds_tested
        mean_qa = sum(qa_costs) / seeds_tested
        mean_saved = mean_blind - mean_qa
        pct_improvement = (mean_saved / mean_blind) * 100.0
        win_count = sum(1 for b, q in zip(blind_costs, qa_costs) if q <= b)

        print("-" * 70)
        print(f"BENCHMARK STATISTICAL SUMMARY ({seeds_tested} Seeds):")
        print(f"  • Mean Queue-Blind Completion Time : {mean_blind:.2f} mins")
        print(f"  • Mean Queue-Aware Completion Time : {mean_qa:.2f} mins")
        print(f"  • Net Mean Time Saved              : {mean_saved:.2f} mins ({pct_improvement:+.2f}%)")
        print(f"  • Win Rate (QA <= Blind)           : {win_count}/{seeds_tested} ({win_count/seeds_tested*100:.1f}%)")
        print("=" * 70)

        # Acceptance Criteria Assertion
        self.assertLess(
            mean_qa,
            mean_blind,
            f"Queue-aware planner failed to achieve lower mean completion time: {mean_qa} >= {mean_blind}"
        )
        print("🎉 ACCEPTANCE CRITERION VERIFIED: Queue-aware planner achieves lower total completion time!")


if __name__ == "__main__":
    unittest.main()
