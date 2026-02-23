"""
ab_test.py - Statistical A/B testing utilities.
Supports z-test, chi-square, t-test, Mann-Whitney U, and sequential testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ABTest:
    """
    Comprehensive A/B testing class supporting:
    - Two-proportion z-test (for conversion rates)
    - Chi-square test
    - Mann-Whitney U test (non-parametric)
    - Sequential probability ratio test (SPRT)
    """

    def __init__(
        self,
        control: pd.Series,
        treatment: pd.Series,
        alpha: float = 0.05,
        power: float = 0.8,
    ):
        """
        Parameters
        ----------
        control : pd.Series  - Outcomes for control group (0/1 or continuous)
        treatment : pd.Series - Outcomes for treatment group
        alpha : float         - Significance level (Type I error)
        power : float         - Desired statistical power (1 - Type II error)
        """
        self.control = control.dropna()
        self.treatment = treatment.dropna()
        self.alpha = alpha
        self.power = power

    # ──────────────────────────────────────────────────────────────
    # DESCRIPTIVE STATS
    # ──────────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        """Return descriptive statistics for both groups."""
        rows = []
        for name, series in [("Control", self.control), ("Treatment", self.treatment)]:
            rows.append({
                "Group": name,
                "N": len(series),
                "Mean": series.mean(),
                "Std": series.std(),
                "Conversion Rate": series.mean() if series.isin([0, 1]).all() else np.nan,
                "Min": series.min(),
                "Median": series.median(),
                "Max": series.max(),
            })
        return pd.DataFrame(rows)

    # ──────────────────────────────────────────────────────────────
    # SAMPLE SIZE CALCULATOR
    # ──────────────────────────────────────────────────────────────
    def required_sample_size(self, baseline_rate: float, mde: float) -> int:
        """
        Calculate minimum detectable effect sample size.
        
        Parameters
        ----------
        baseline_rate : float - Baseline conversion rate
        mde           : float - Minimum detectable effect (relative lift, e.g. 0.05 for 5%)
        """
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta  = stats.norm.ppf(self.power)
        pooled  = (p1 + p2) / 2
        n = (2 * pooled * (1 - pooled) * (z_alpha + z_beta) ** 2) / (p1 - p2) ** 2
        return int(np.ceil(n))

    # ──────────────────────────────────────────────────────────────
    # STATISTICAL TESTS
    # ──────────────────────────────────────────────────────────────
    def z_test(self) -> dict:
        """Two-proportion z-test for conversion rates (binary outcomes)."""
        n_c = len(self.control)
        n_t = len(self.treatment)
        conv_c = self.control.sum()
        conv_t = self.treatment.sum()
        p_c = conv_c / n_c
        p_t = conv_t / n_t
        p_pool = (conv_c + conv_t) / (n_c + n_t)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
        z = (p_t - p_c) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        lift = (p_t - p_c) / p_c * 100 if p_c > 0 else 0
        ci_lower = (p_t - p_c) - 1.96 * se
        ci_upper = (p_t - p_c) + 1.96 * se
        return {
            "test": "Two-Proportion Z-Test",
            "control_rate": round(p_c, 4),
            "treatment_rate": round(p_t, 4),
            "lift_pct": round(lift, 2),
            "z_statistic": round(z, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < self.alpha,
            "ci_diff_95": (round(ci_lower, 4), round(ci_upper, 4)),
        }

    def chi_square_test(self) -> dict:
        """Chi-square test of independence."""
        conv_c = int(self.control.sum())
        non_c  = len(self.control) - conv_c
        conv_t = int(self.treatment.sum())
        non_t  = len(self.treatment) - conv_t
        contingency = np.array([[conv_c, non_c], [conv_t, non_t]])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        return {
            "test": "Chi-Square Test",
            "chi2_statistic": round(chi2, 4),
            "p_value": round(p_value, 6),
            "degrees_of_freedom": dof,
            "significant": p_value < self.alpha,
        }

    def mann_whitney_test(self) -> dict:
        """Mann-Whitney U test (non-parametric, no normality assumption)."""
        u_stat, p_value = stats.mannwhitneyu(
            self.control, self.treatment, alternative="two-sided"
        )
        return {
            "test": "Mann-Whitney U Test",
            "u_statistic": round(u_stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < self.alpha,
        }

    def t_test(self) -> dict:
        """Welch's t-test (continuous outcomes)."""
        t_stat, p_value = stats.ttest_ind(self.control, self.treatment, equal_var=False)
        return {
            "test": "Welch's T-Test",
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < self.alpha,
            "mean_diff": round(self.treatment.mean() - self.control.mean(), 4),
        }

    # ──────────────────────────────────────────────────────────────
    # SEQUENTIAL TEST (SPRT)
    # ──────────────────────────────────────────────────────────────
    def sprt(self, p0: float, p1: float) -> dict:
        """
        Sequential Probability Ratio Test.
        
        Parameters
        ----------
        p0 : float - Null hypothesis conversion rate
        p1 : float - Alternative hypothesis conversion rate (MDE)
        """
        A = (1 - self.alpha) / (1 - self.power)
        B = self.alpha / self.power
        llr = 0
        decisions = []
        for obs in pd.concat([self.control, self.treatment]):
            if obs == 1:
                llr += np.log(p1 / p0)
            else:
                llr += np.log((1 - p1) / (1 - p0))
            if llr >= np.log(A):
                decisions.append("accept_H1")
            elif llr <= np.log(B):
                decisions.append("reject_H1")
            else:
                decisions.append("continue")
        final = decisions[-1] if decisions else "continue"
        return {
            "test": "SPRT",
            "final_decision": final,
            "log_likelihood_ratio": round(llr, 4),
            "boundary_A": round(np.log(A), 4),
            "boundary_B": round(np.log(B), 4),
        }

    # ──────────────────────────────────────────────────────────────
    # FULL REPORT
    # ──────────────────────────────────────────────────────────────
    def full_report(self) -> dict:
        """Run all applicable tests and return combined results."""
        is_binary = set(self.control.unique()).issubset({0, 1}) and \
                    set(self.treatment.unique()).issubset({0, 1})
        report = {"summary": self.summary().to_dict(orient="records")}
        if is_binary:
            report["z_test"] = self.z_test()
            report["chi_square"] = self.chi_square_test()
        report["mann_whitney"] = self.mann_whitney_test()
        report["t_test"] = self.t_test()
        return report


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    ctrl = pd.Series(np.random.binomial(1, 0.10, 1000))
    treat = pd.Series(np.random.binomial(1, 0.12, 1000))
    ab = ABTest(ctrl, treat)
    report = ab.full_report()
    for k, v in report.items():
        print(f"\n{k}:\n{v}")