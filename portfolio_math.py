import numpy as np

# Constants based on the given assumptions
V_N = 5_000_000_000  # FDV of native tokens
V_B = 5_000_000_000  # value of Bitcoin holdings
V_T = V_N + V_B  # total portfolio value

# Weights of each asset in the portfolio
w_N = V_N / V_T
w_B = V_B / V_T

# Staking APR variables
TS = 10_000_000_000  # total supply of tokens
I = 0.10  # interest rate
CT = 0.02  # commission taken
BT = 6_000_000_000  # base token amount
BTR = BT / TS  # base token ratio
VC = 0.05  # validator commission
TAB = 6_311_520_000  # total active balance
AAB = 6_000_000_000  # active account balance

# Market returns
R_Price_N = 0.05  # expected price return on native token
R_Price_B = 0.08  # expected price return on Bitcoin
R_f = 0.02  # risk-free rate

# Standard deviations and correlation
sigma_N = 0.25  # native token standard deviation
sigma_B = 0.20  # Bitcoin standard deviation
rho_NB = 0.3  # correlation coefficient


def main():
    # Calculating Nominal APR
    APR_Nominal = (I * (1 - CT)) / BTR
    # Calculating Adjustment Factor
    Adjustment_Factor = AAB / TAB
    # Calculating Actual APR
    APR_Actual = APR_Nominal * Adjustment_Factor
    # Calculating Final APR
    APR_Final = APR_Actual * (1 - VC)
    # Total Expected Return on Native Token
    R_N = R_Price_N + APR_Final

    # Expected Portfolio Return
    R_P = w_N * R_N + w_B * R_Price_B
    Excess_Return_P = R_P - R_f

    # Covariance and Portfolio Variance/Standard Deviation
    sigma_NB = rho_NB * sigma_N * sigma_B
    portfolio_variance = (
        (w_N**2 * sigma_N**2) + (w_B**2 * sigma_B**2) + (2 * w_N * w_B * sigma_NB)
    )
    sigma_P = np.sqrt(portfolio_variance)

    # Sharpe Ratios
    Excess_Return_N = R_N - R_f
    Sharpe_Single = Excess_Return_N / sigma_N
    Sharpe_Joint = Excess_Return_P / sigma_P

    # Results dictionary to display
    results = {
        "Nominal APR (%)": APR_Nominal * 100,
        "Adjustment Factor": Adjustment_Factor,
        "Actual APR (%)": APR_Actual * 100,
        "Final APR (%)": APR_Final * 100,
        "Total Expected Return on Native Token (%)": R_N * 100,
        "Native token Standard Deviation (%)": sigma_N * 100,
        "Expected Portfolio Return (%)": R_P * 100,
        "Excess Return over Risk-Free Rate (%)": Excess_Return_P * 100,
        "Portfolio Variance (%)": portfolio_variance * 100,
        "Portfolio Standard Deviation (%)": sigma_P * 100,
        "Sharpe Ratio (Single Asset)": Sharpe_Single,
        "Sharpe Ratio (Portfolio)": Sharpe_Joint,
    }
    import pandas as pd

    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    print(results_df)


# import ace_tools as tools; tools.display_dataframe_to_user(name="Staking and Portfolio Simulation Results", dataframe=results)
if __name__ == "__main__":
    main()
