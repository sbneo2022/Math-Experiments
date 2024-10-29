from dataclasses import dataclass
from abc import ABC, abstractmethod
import math


@dataclass
class Token:
    name: str
    stake_amount: float  # S_N or S_B
    price: float  # p_N or p_B
    volatility: float  # σ_N or σ_B

    def value(self) -> float:
        """Calculate the total value of the token stake."""
        return self.stake_amount * self.price

    def variance(self) -> float:
        """Calculate the variance of the token stake."""
        return (self.value() * self.volatility) ** 2


class Portfolio(ABC):
    @abstractmethod
    def variance(self) -> float:
        """Calculate the variance of the portfolio."""
        pass

    def standard_deviation(self) -> float:
        """Calculate the standard deviation of the portfolio."""
        return math.sqrt(self.variance())


@dataclass
class SinglePortfolio(Portfolio):
    token: Token

    def variance(self) -> float:
        return self.token.variance()


@dataclass
class JointPortfolio(Portfolio):
    tokens: list  # List of Token instances

    def variance(self) -> float:
        # Assuming negligible correlation between assets
        return sum(token.variance() for token in self.tokens)


@dataclass
class InterestCalculator:
    native_token: Token
    bitcoin_token: Token
    native_apy: float  # r_N
    vol_multiplier: float  # V_m

    def calculate(self) -> dict:
        """Perform the calculations and return the results."""
        # Create portfolios
        single_portfolio = SinglePortfolio(token=self.native_token)
        joint_portfolio = JointPortfolio(tokens=[self.native_token, self.bitcoin_token])

        # Calculate standard deviations
        sd_N = single_portfolio.standard_deviation()
        sd_joint = joint_portfolio.standard_deviation()

        # Interest payments
        V_N = self.native_token.value()
        interest_N = V_N * self.native_apy

        # Adjusted interest payment for joint portfolio
        interest_joint = interest_N * (sd_joint / sd_N)

        # Compute the spread
        spread = interest_N - interest_joint

        # Spread rate
        spread_rate = spread / V_N

        # Compute the Bitcoin APY
        bitcoin_apy = (self.native_apy - spread_rate) / self.vol_multiplier

        # Compile results
        results = {
            "Native Token Value": V_N,
            "Bitcoin Value": self.bitcoin_token.value(),
            "Variance Native": single_portfolio.variance(),
            "Variance Bitcoin": self.bitcoin_token.variance(),
            "Variance Joint": joint_portfolio.variance(),
            "SD Native": sd_N,
            "SD Joint": sd_joint,
            "Interest Native": interest_N,
            "Interest Joint": interest_joint,
            "Spread": spread,
            "Spread Rate": spread_rate,
            "Bitcoin APY": bitcoin_apy,
        }
        return results


def main():
    # Input parameters
    S_N = 1000  # Native token stake amount
    p_N = 10  # Price per unit of native token
    σ_N = 0.80  # Volatility of native token

    S_B = 0.5  # Bitcoin stake amount
    p_B = 20000  # Price per Bitcoin
    σ_B = 0.20  # Volatility of Bitcoin

    r_N = 0.10  # Native token APY
    V_m = 4  # Volatility multiplier

    # Create token instances
    native_token = Token(
        name="Native Token", stake_amount=S_N, price=p_N, volatility=σ_N
    )

    bitcoin_token = Token(name="Bitcoin", stake_amount=S_B, price=p_B, volatility=σ_B)

    # Create the interest calculator
    calculator = InterestCalculator(
        native_token=native_token,
        bitcoin_token=bitcoin_token,
        native_apy=r_N,
        vol_multiplier=V_m,
    )

    # Perform the calculation
    results = calculator.calculate()

    # Display the results
    print("\n--- Calculation Results ---\n")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
