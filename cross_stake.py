from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ChainParams:
    total_staked_c: float  # S_C: Total staked on consumer chain
    cross_staked_p: float  # S_P: Total staked from provider chain
    q: float  # q: Max fraction of voting power provider can control


@dataclass
class MultiChainParams:
    total_staked_c: float  # S_C: Total staked on consumer chain
    cross_staked_p: Dict[str, float]  # S_P_i: Staked from each provider chain
    q: Dict[str, float]  # q_i: Max fraction per provider


@dataclass
class SimulationResult:
    alpha: float
    beta: Dict[str, float]
    voting_power_distribution: Dict[str, float]


class BaseStakingSimulator(ABC):
    @abstractmethod
    def compute_alpha(self):
        pass

    @abstractmethod
    def compute_beta(self):
        pass

    @abstractmethod
    def run_simulation(self):
        pass


class SingleProviderChainSimulator(BaseStakingSimulator):
    def __init__(self, params: ChainParams):
        self.params = params
        self.alpha = 0.0
        self.beta = 0.0

    def compute_alpha(self):
        if self.params.cross_staked_p == 0:
            self.alpha = 1 / self.params.total_staked_c
        else:
            crossing_point = (
                self.params.q / (1 - self.params.q) * self.params.total_staked_c
            )
            if self.params.cross_staked_p <= crossing_point:
                self.alpha = 1 / (
                    self.params.total_staked_c + self.params.cross_staked_p
                )
            else:
                self.alpha = (1 - self.params.q) / self.params.total_staked_c
        return self.alpha

    def compute_beta(self):
        if self.params.cross_staked_p == 0:
            self.beta = 0
        else:
            crossing_point = (
                self.params.q / (1 - self.params.q) * self.params.total_staked_c
            )
            if self.params.cross_staked_p <= crossing_point:
                self.beta = self.alpha
            else:
                self.beta = self.params.q / self.params.cross_staked_p
        return self.beta

    def run_simulation(self):
        alpha = self.compute_alpha()
        beta = self.compute_beta()
        voting_power_distribution = {
            "ConsumerChain": alpha * self.params.total_staked_c,
            "ProviderChain": beta * self.params.cross_staked_p,
        }
        return SimulationResult(
            alpha=alpha,
            beta={"ProviderChain": beta},
            voting_power_distribution=voting_power_distribution,
        )


class MultiProviderChainSimulator(BaseStakingSimulator):
    def __init__(self, params: MultiChainParams):
        self.params = params
        self.alpha = 0.0
        self.beta = {p: 0.0 for p in params.cross_staked_p}
        self.crossing_point_met = {p: False for p in params.cross_staked_p}

    def compute_alpha(self):
        P = set(
            self.params.cross_staked_p.keys()
        )  # Initially, all providers are not capped
        while True:
            sum_q_not_in_p = sum(
                self.params.q[p] for p in self.params.cross_staked_p if p not in P
            )
            sum_staked_in_p = sum(self.params.cross_staked_p[p] for p in P)

            new_alpha = (1 - sum_q_not_in_p) / (
                self.params.total_staked_c + sum_staked_in_p
            )

            # Check which providers are capped
            newly_capped = {
                p
                for p in P
                if new_alpha * self.params.cross_staked_p[p] > self.params.q[p]
            }

            if not newly_capped:
                break  # Exit if no new caps

            P -= newly_capped  # Update P with newly capped providers

        self.alpha = new_alpha
        return self.alpha

    def compute_beta(self):
        for p in self.params.cross_staked_p:
            if p in self.crossing_point_met:
                self.beta[p] = self.alpha
            else:
                self.beta[p] = min(
                    self.alpha, self.params.q[p] / self.params.cross_staked_p[p]
                )
        return self.beta

    def run_simulation(self):
        alpha = self.compute_alpha()
        beta = self.compute_beta()
        voting_power_distribution = {
            "ConsumerChain": alpha * self.params.total_staked_c,
        }
        for p in self.params.cross_staked_p:
            voting_power_distribution[p] = beta[p] * self.params.cross_staked_p[p]

        return SimulationResult(
            alpha=alpha, beta=beta, voting_power_distribution=voting_power_distribution
        )


# Example usage
if __name__ == "__main__":
    # Single provider chain example
    single_params = ChainParams(total_staked_c=1000, cross_staked_p=500, q=0.4)
    single_simulator = SingleProviderChainSimulator(single_params)
    single_result = single_simulator.run_simulation()
    print(f"Single Provider Chain Simulation:\n{single_result}\n")

    # Multiple provider chains example
    multi_params = MultiChainParams(
        total_staked_c=1000,
        cross_staked_p={"P1": 300, "P2": 200},
        q={"P1": 0.3, "P2": 0.2},
    )
    multi_simulator = MultiProviderChainSimulator(multi_params)
    multi_result = multi_simulator.run_simulation()
    print(f"Multiple Provider Chains Simulation:\n{multi_result}")
