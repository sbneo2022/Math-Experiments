from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class RewardFunctionData:
    """Data class to store reward function parameters."""

    name: str
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    k: float = 1.0
    gamma: float = 0.5


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    def __init__(self, data: RewardFunctionData):
        self.data = data

    @abstractmethod
    def compute_reward(self, s_btc, s_bbn):
        """Compute the reward based on stakes s_btc and s_bbn."""
        pass

    def simulate(self, s_btc_range, s_bbn_range):
        """Simulate the reward function over given ranges of stakes."""
        rewards = np.zeros((len(s_btc_range), len(s_bbn_range)))
        for i, s_btc in enumerate(s_btc_range):
            for j, s_bbn in enumerate(s_bbn_range):
                rewards[i, j] = self.compute_reward(s_btc, s_bbn)
        return rewards

    def plot_rewards(self, s_btc_range, s_bbn_range):
        """Plot the reward function as a heatmap."""
        rewards = self.simulate(s_btc_range, s_bbn_range)
        S_BTC, S_BBN = np.meshgrid(s_btc_range, s_bbn_range, indexing="ij")
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(S_BTC, S_BBN, rewards, levels=50, cmap="viridis")
        plt.colorbar(cp)
        plt.xlabel("s_btc (BTC Staked)")
        plt.ylabel("s_bbn (BBN Staked)")
        plt.title(f"Reward Function: {self.data.name}")
        plt.show()


class LinearRewardFunction(RewardFunction):
    """Linear reward function: R(s_btc, s_bbn) = a * s_btc + b * s_bbn."""

    def compute_reward(self, s_btc, s_bbn):
        return self.data.a * s_btc + self.data.b * s_bbn


class AdditiveCrossTermRewardFunction(RewardFunction):
    """Additive reward with cross term: R(s_btc, s_bbn) = a * s_btc + b * s_bbn + c * s_btc * s_bbn."""

    def compute_reward(self, s_btc, s_bbn):
        return self.data.a * s_btc + self.data.b * s_bbn + self.data.c * s_btc * s_bbn


class ConcaveInteractionRewardFunction(RewardFunction):
    """Modified reward with concave interaction: R(s_btc, s_bbn) = a * s_btc * f(s_bbn) + b * s_bbn."""

    def f(self, s_bbn):
        """Concave, increasing function f(s_bbn)."""
        return 1 + (self.data.c * s_bbn) / (1 + s_bbn)

    def compute_reward(self, s_btc, s_bbn):
        return self.data.a * s_btc * self.f(s_bbn) + self.data.b * s_bbn


class GeometricMeanRewardFunction(RewardFunction):
    """Geometric mean-based reward: R(s_btc, s_bbn) = a * s_btc^gamma * s_bbn^(1 - gamma)."""

    def compute_reward(self, s_btc, s_bbn):
        return self.data.a * (s_btc**self.data.gamma) * (s_bbn ** (1 - self.data.gamma))


class MinFunctionReward(RewardFunction):
    """Proposed reward function: R(s_btc, s_bbn) = a * s_btc + b * s_bbn + c * min(s_btc, k * s_bbn)."""

    def compute_reward(self, s_btc, s_bbn):
        min_term = min(s_btc, self.data.k * s_bbn)
        return self.data.a * s_btc + self.data.b * s_bbn + self.data.c * min_term


# Example usage
def main():
    # Define stake ranges
    s_btc_range = np.linspace(0, 10, 100)  # BTC stakes from 0 to 10
    s_bbn_range = np.linspace(0, 10, 100)  # BBN stakes from 0 to 10

    # Linear Reward Function
    linear_data = RewardFunctionData(name="Linear Reward Function", a=1.0, b=1.0)
    linear_reward = LinearRewardFunction(data=linear_data)
    linear_reward.plot_rewards(s_btc_range, s_bbn_range)

    # Additive Cross Term Reward Function
    cross_data = RewardFunctionData(
        name="Additive Cross Term Reward Function", a=1.0, b=1.0, c=0.1
    )
    cross_reward = AdditiveCrossTermRewardFunction(data=cross_data)
    cross_reward.plot_rewards(s_btc_range, s_bbn_range)

    # Concave Interaction Reward Function
    concave_data = RewardFunctionData(
        name="Concave Interaction Reward Function", a=1.0, b=1.0, c=1.0
    )
    concave_reward = ConcaveInteractionRewardFunction(data=concave_data)
    concave_reward.plot_rewards(s_btc_range, s_bbn_range)

    # Geometric Mean-Based Reward Function
    geometric_data = RewardFunctionData(
        name="Geometric Mean Reward Function", a=1.0, gamma=0.5
    )
    geometric_reward = GeometricMeanRewardFunction(data=geometric_data)
    geometric_reward.plot_rewards(s_btc_range, s_bbn_range)

    # Proposed Min Function Reward
    min_data = RewardFunctionData(
        name="Proposed Min Function Reward", a=1.0, b=1.0, c=1.0, k=1.0
    )
    min_reward = MinFunctionReward(data=min_data)
    min_reward.plot_rewards(s_btc_range, s_bbn_range)


if __name__ == "__main__":
    main()
