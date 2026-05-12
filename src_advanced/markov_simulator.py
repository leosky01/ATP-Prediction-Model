"""
Tennis match simulator using Markov chain model.

Based on Newton & Keller (2005) "Probability of Winning at Tennis"
and Barnett & Clarke (2005).

The tennis scoring hierarchy is: point → game → set → match.
Given two parameters:
    p_A = probability player A wins a point on their own serve
    p_B = probability player B wins a point on their own serve

We can analytically compute:
    - Game hold probability for each player
    - Set win probability
    - Match win probability
    - Full distribution of total games
    - P(total games > threshold) for O/U betting

Usage:
    from markov_simulator import TennisSimulator
    sim = TennisSimulator(p_A=0.65, p_B=0.62)
    result = sim.simulate_match(best_of=3)
    print(result['p_over_22_5'])  # probability of going over 22.5 games
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class TennisSimulator:
    """
    Analytical tennis match simulator using Markov chains.

    Parameters
    ----------
    p_A : float
        Probability that player A wins a point on A's serve (typically 0.55-0.75 ATP)
    p_B : float
        Probability that player B wins a point on B's serve
    """

    def __init__(self, p_A: float, p_B: float):
        self.p_A = np.clip(p_A, 0.01, 0.99)
        self.p_B = np.clip(p_B, 0.01, 0.99)
        # Game hold probabilities (Newton & Keller formula)
        self.g_A = self._game_hold_prob(self.p_A)
        self.g_B = self._game_hold_prob(self.p_B)

    @staticmethod
    def _game_hold_prob(p: float) -> float:
        """
        Probability of holding serve given point-win probability p.

        Uses the closed-form solution from Newton & Keller (2005), eq. 5:
            G(p) = p^4 * (15 - 4q - 10q^2) / (1 - 2pq)
        where q = 1-p.

        This covers regular games (win by 2 after deuce).
        """
        q = 1.0 - p
        # Regular game: win 4 points with margin >= 2
        # P(win without deuce) = p^4 * (1 + 4q + 10q^2)
        no_deuce = p**4 * (1 + 4 * q + 10 * q**2)
        # P(reach deuce) = p^4 * 10q^2 * ... actually use the full formula
        # After deuce, it's a geometric series: need to win by 2
        # P(hold | deuce) = p^2 / (1 - 2*p*q)  [need p != 0.5]
        deuce_factor = 20 * (p * q)**3  # probability of reaching deuce
        if abs(1 - 2 * p * q) > 1e-10:
            hold = no_deuce + deuce_factor * (p**2) / (1 - 2 * p * q)
        else:
            # p = q = 0.5: hold prob = 0.5
            hold = no_deuce + deuce_factor * 0.5
        return np.clip(hold, 0.0, 1.0)

    @staticmethod
    def _tiebreak_game_hold(
        p_server: float, p_returner_serve: float, server_is_A: bool
    ) -> float:
        """
        Probability that the serving player wins a point in a tiebreak.
        In a tiebreak, serve alternates: A, B, B, A, A, B, B, A, ...
        But for individual point probability, we just use who's serving.
        """
        return p_server

    def _simulate_set_game_by_game(
        self, A_serves_first: bool = True
    ) -> Dict:
        """
        Simulate a single set using game-level Markov chain.

        States: (games_A, games_B) with serving alternation.
        Returns probabilities of each score and total games.

        Returns dict with:
            p_A_wins: probability player A wins the set
            total_games_dist: dict {n_games: probability}
        """
        # State: (games_A, games_B) → probability
        # We track who serves: alternates starting from A_serves_first
        # Games 0,2,4,6,... first player serves (if A_serves_first)
        # Games 1,3,5,... second player serves

        # Use DP: states are (i, j) = games won by A, games won by B
        # Maximum before tiebreak: 6-6 → tiebreak (13 total games)
        # Normal win: first to 7 with margin >= 2 (but in games, first to 6 with margin >= 2, or 7-5)

        # Transition probabilities alternate between g_A (A serves) and g_B (B serves)
        # Game number n (0-indexed): if n%2==0 → first server serves, else second serves

        max_normal = 12  # 6-6 → tiebreak

        # DP table: dp[i][j] = probability of reaching state (i,j)
        dp = np.zeros((8, 8))
        dp[0][0] = 1.0

        total_games_dist = {}
        p_A_wins_set = 0.0
        p_B_wins_set = 0.0

        for total_games in range(1, 13):
            new_dp = np.zeros((8, 8))
            for i in range(min(total_games + 1, 7)):
                j = total_games - 1 - i  # previous total was total_games - 1
                if j < 0 or j > 6:
                    continue
                if dp[i][j] == 0:
                    continue

                # Who serves this game?
                game_num = i + j  # current game number (0-indexed)
                if A_serves_first:
                    a_serves = (game_num % 2 == 0)
                else:
                    a_serves = (game_num % 2 == 1)

                if a_serves:
                    p_A_wins_game = self.g_A
                else:
                    p_A_wins_game = 1 - self.g_B  # B serves, A wins = 1 - g_B

                # A wins this game
                new_i, new_j = i + 1, j
                # Check if set is over
                if new_i >= 6 and new_j <= 4 and (new_i - new_j) >= 2:
                    # A wins set normally
                    games = new_i + new_j
                    p_A_wins_set += dp[i][j] * p_A_wins_game
                    total_games_dist[games] = total_games_dist.get(games, 0) + dp[i][j] * p_A_wins_game
                elif new_i == 7 and new_j == 5:
                    # A wins 7-5
                    p_A_wins_set += dp[i][j] * p_A_wins_game
                    total_games_dist[12] = total_games_dist.get(12, 0) + dp[i][j] * p_A_wins_game
                elif new_i <= 6 and new_j <= 6:
                    new_dp[min(new_i, 6)][min(new_j, 6)] += dp[i][j] * p_A_wins_game

                # B wins this game
                new_i, new_j = i, j + 1
                if new_j >= 6 and new_i <= 4 and (new_j - new_i) >= 2:
                    # B wins set normally
                    games = new_i + new_j
                    p_B_wins_set += dp[i][j] * (1 - p_A_wins_game)
                    total_games_dist[games] = total_games_dist.get(games, 0) + dp[i][j] * (1 - p_A_wins_game)
                elif new_j == 7 and new_i == 5:
                    # B wins 5-7
                    p_B_wins_set += dp[i][j] * (1 - p_A_wins_game)
                    total_games_dist[12] = total_games_dist.get(12, 0) + dp[i][j] * (1 - p_A_wins_game)
                elif new_i <= 6 and new_j <= 6:
                    new_dp[min(new_i, 6)][min(new_j, 6)] += dp[i][j] * (1 - p_A_wins_game)

            dp = new_dp

        # Handle tiebreak (6-6)
        if dp[6][6] > 0:
            p_tb = dp[6][6]
            # Tiebreak: 13 total games
            # Simplification: compute tiebreak win probability analytically
            p_A_wins_tb = self._tiebreak_prob(A_serves_first)
            p_A_wins_set += p_tb * p_A_wins_tb
            p_B_wins_set += p_tb * (1 - p_A_wins_tb)
            total_games_dist[13] = total_games_dist.get(13, 0) + p_tb

        return {
            "p_A_wins": p_A_wins_set,
            "p_B_wins": p_B_wins_set,
            "total_games_dist": total_games_dist,
        }

    def _tiebreak_prob(self, A_serves_first: bool = True) -> float:
        """
        Probability that player A wins a tiebreak.

        Tiebreak serving pattern: A, B, B, A, A, B, B, A, ...
        (if A serves first). Points alternate in pairs after the first point.
        First to 7 with margin >= 2, or continue until margin = 2.
        """
        # Point-by-point simulation up to 7-7, then geometric series for deuce
        # States: (pts_A, pts_B)
        # Serving pattern: point 1: A, points 2-3: B, points 4-5: A, points 6-7: B, ...

        def point_server(pt_num: int) -> str:
            """Who serves point pt_num (1-indexed)?"""
            if pt_num == 1:
                return 'A' if A_serves_first else 'B'
            # After point 1, groups of 2: points 2-3, 4-5, 6-7, ...
            group = (pt_num - 2) // 2  # 0-indexed group
            if group % 2 == 0:
                return 'B' if A_serves_first else 'A'
            else:
                return 'A' if A_serves_first else 'B'

        def p_A_wins_point(pt_num: int) -> float:
            server = point_server(pt_num)
            if server == 'A':
                return self.p_A
            else:
                return 1 - self.p_B

        # DP: simulate points up to potential 7-7
        # dp[pts_A][pts_B] = probability of reaching this score
        max_pts = 8
        dp = np.zeros((max_pts, max_pts))
        dp[0][0] = 1.0

        p_A_wins_tb = 0.0

        for total_pts in range(1, 15):
            new_dp = np.zeros((max_pts, max_pts))
            for pa in range(min(total_pts, max_pts)):
                pb = total_pts - 1 - pa
                if pb < 0 or pb >= max_pts:
                    continue
                if dp[pa][pb] == 0:
                    continue

                pp = p_A_wins_point(pa + pb + 1)  # 1-indexed point number

                # A wins point
                na, nb = pa + 1, pb
                if na >= 7 and (na - nb) >= 2:
                    p_A_wins_tb += dp[pa][pb] * pp
                elif na < max_pts and nb < max_pts:
                    new_dp[na][nb] += dp[pa][pb] * pp

                # B wins point
                na, nb = pa, pb + 1
                if nb >= 7 and (nb - na) >= 2:
                    pass  # B wins, we track only A's prob
                elif na < max_pts and nb < max_pts:
                    new_dp[na][nb] += dp[pa][pb] * (1 - pp)

            dp = new_dp

        # Handle extended tiebreak (7-7, 8-8, ...) with geometric series
        # At 7-7 (or any tied score >= 6-6), probability of A winning
        # depends on the serving pattern. For simplicity, compute
        # the probability of A winning from each tied state.
        for tied_score in range(6, max_pts):
            if dp[tied_score][tied_score] == 0:
                continue
            # From tied score s-s, need to win 2 consecutive points (in a pair)
            # or go back to tied. This is a geometric series.
            # Next 2 points: who serves?
            pt1 = 2 * tied_score + 1  # 1-indexed
            pt2 = 2 * tied_score + 2
            p1 = p_A_wins_point(pt1)  # P(A wins point 1)
            p2 = p_A_wins_point(pt2)  # P(A wins point 2)

            # P(A wins both) = p1*p2 → A wins tiebreak
            # P(B wins both) = (1-p1)*(1-p2) → B wins tiebreak
            # P(split) = 1 - p1*p2 - (1-p1)*(1-p2) → back to tied
            p_A_win_pair = p1 * p2
            p_B_win_pair = (1 - p1) * (1 - p2)
            p_split = 1 - p_A_win_pair - p_B_win_pair

            if p_split < 1 - 1e-10:
                # Geometric series: P(A wins) = p_A / (1 - p_split)
                p_A_from_tied = p_A_win_pair / (1 - p_split)
            else:
                p_A_from_tied = 0.5  # edge case

            p_A_wins_tb += dp[tied_score][tied_score] * p_A_from_tied

        return p_A_wins_tb

    def simulate_match(
        self,
        best_of: int = 3,
        A_serves_first_set: bool = True,
        n_simulations: int = 0,
    ) -> Dict:
        """
        Simulate a full match and return total games distribution.

        Parameters
        ----------
        best_of : int
            3 for best-of-3, 5 for best-of-5
        A_serves_first_set : bool
            Whether player A serves first in the first set
            (subsequent sets alternate first server)
        n_simulations : int
            If > 0, use Monte Carlo instead of analytical.
            0 means use analytical (exact) computation.

        Returns
        -------
        dict with keys:
            expected_total_games: float
            total_games_dist: dict {n_games: probability}
            p_over: dict {threshold: probability}
            p_A_wins_match: float
        """
        if n_simulations > 0:
            return self._simulate_monte_carlo(best_of, A_serves_first_set, n_simulations)

        # Analytical approach: enumerate all possible match outcomes
        # For BO3: A wins in 2 sets, or B wins in 2 sets, or goes to 3rd set
        # For BO5: enumerate all 2-set, 3-set, 4-set, 5-set outcomes

        n_sets_to_win = (best_of + 1) // 2  # 2 for BO3, 3 for BO5

        # Get single set distribution (A serves first)
        set_A_first = self._simulate_set_game_by_game(A_serves_first=True)
        set_B_first = self._simulate_set_game_by_game(A_serves_first=False)

        # Expected total games from a single set
        def expected_games_from_dist(dist):
            return sum(n * p for n, p in dist.items())

        # Compute match-level distribution by enumerating all possible score paths
        # State: (sets_A, sets_B, total_games_so_far, next_server_is_A)
        from collections import defaultdict

        match_dist = defaultdict(float)  # total_games -> probability
        p_A_wins = 0.0

        # Use DP over set outcomes
        # state: (sets_A, sets_B) → (prob, {total_games: prob})
        # This gets complex for BO5, so we use a recursive approach
        def enumerate_paths(
            sets_A: int, sets_B: int,
            prob_so_far: float,
            games_dist_so_far: Dict[int, float],
            A_serves_next: bool,
        ) -> None:
            """Recursively enumerate all match completion paths."""
            if sets_A == n_sets_to_win or sets_B == n_sets_to_win:
                # Match is over
                for total_g, p in games_dist_so_far.items():
                    match_dist[total_g] += prob_so_far * p
                if sets_A == n_sets_to_win:
                    nonlocal p_A_wins
                    p_A_wins += prob_so_far
                return

            # Play next set
            set_result = set_A_first if A_serves_next else set_B_first
            p_A_wins_set = set_result["p_A_wins"]
            p_B_wins_set = set_result["p_B_wins"]
            set_game_dist = set_result["total_games_dist"]

            # Combine game distributions
            new_dist_A = defaultdict(float)  # if A wins set
            new_dist_B = defaultdict(float)  # if B wins set

            for existing_g, existing_p in games_dist_so_far.items():
                for set_g, set_p in set_game_dist.items():
                    total = existing_g + set_g
                    new_dist_A[total] += existing_p * set_p
                    new_dist_B[total] += existing_p * set_p

            # A wins this set
            enumerate_paths(
                sets_A + 1, sets_B,
                prob_so_far * p_A_wins_set,
                dict(new_dist_A),
                not A_serves_next,  # alternate first server
            )

            # B wins this set
            enumerate_paths(
                sets_A, sets_B + 1,
                prob_so_far * p_B_wins_set,
                dict(new_dist_B),
                not A_serves_next,
            )

        enumerate_paths(0, 0, 1.0, {0: 1.0}, A_serves_first_set)

        expected_total = sum(n * p for n, p in match_dist.items())
        total_prob = sum(match_dist.values())

        # Compute P(total > threshold) for standard thresholds
        p_over = {}
        for threshold in [20.5, 21.5, 22.5, 23.5, 24.5,
                          30.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5]:
            p_over[threshold] = sum(
                p for n, p in match_dist.items() if n > threshold
            )

        return {
            "expected_total_games": expected_total,
            "total_games_dist": dict(sorted(match_dist.items())),
            "p_over": p_over,
            "p_A_wins_match": p_A_wins,
            "total_prob": total_prob,  # should be ~1.0
            "p_A": self.p_A,
            "p_B": self.p_B,
            "g_A": self.g_A,
            "g_B": self.g_B,
        }

    def _simulate_monte_carlo(
        self, best_of: int, A_serves_first: bool, n_sims: int
    ) -> Dict:
        """Monte Carlo simulation for validation."""
        n_sets_to_win = (best_of + 1) // 2
        total_games_list = []
        a_wins = 0

        for _ in range(n_sims):
            sets_A, sets_B = 0, 0
            total_games = 0
            a_serves = A_serves_first

            while sets_A < n_sets_to_win and sets_B < n_sets_to_win:
                # Simulate a set game-by-game
                ga, gb = 0, 0
                a_serves_game = a_serves
                while True:
                    if a_serves_game:
                        a_wins_game = np.random.random() < self.g_A
                    else:
                        a_wins_game = np.random.random() > self.g_B

                    if a_wins_game:
                        ga += 1
                    else:
                        gb += 1

                    # Check set over
                    if ga >= 6 and gb <= 4 and (ga - gb) >= 2:
                        break
                    if gb >= 6 and ga <= 4 and (gb - ga) >= 2:
                        break
                    if ga == 7:
                        break
                    if gb == 7:
                        break
                    if ga == 6 and gb == 6:
                        # Tiebreak
                        if np.random.random() < self._tiebreak_prob(a_serves_game):
                            ga = 7
                        else:
                            gb = 7
                        break

                    a_serves_game = not a_serves_game

                total_games += ga + gb
                if ga > gb:
                    sets_A += 1
                else:
                    sets_B += 1
                a_serves = not a_serves

            total_games_list.append(total_games)
            if sets_A == n_sets_to_win:
                a_wins += 1

        games_arr = np.array(total_games_list)
        p_over = {}
        for th in [20.5, 21.5, 22.5, 23.5, 24.5,
                    30.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5]:
            p_over[th] = float((games_arr > th).mean())

        return {
            "expected_total_games": float(games_arr.mean()),
            "total_games_dist": {int(n): float((games_arr == n).mean())
                                 for n in sorted(set(games_arr.astype(int)))},
            "p_over": p_over,
            "p_A_wins_match": a_wins / n_sims,
            "p_A": self.p_A,
            "p_B": self.p_B,
            "g_A": self.g_A,
            "g_B": self.g_B,
        }


def compute_serve_point_win_rate(
    first_won: int, second_won: int, svpt: int
) -> float:
    """
    Compute serve point win rate from match stats.

    p_serve = (1stWon + 2ndWon) / svpt

    This is the key input to the Markov model.
    """
    if pd.isna(svpt) or svpt == 0 or pd.isna(first_won) or pd.isna(second_won):
        return np.nan
    return (first_won + second_won) / svpt


def compute_game_hold_rate(
    bp_saved: int, bp_faced: int, sv_gms: int
) -> float:
    """
    Compute empirical game hold rate from match stats.

    hold_rate = (sv_gms - (bp_faced - bp_saved)) / sv_gms
              = games held / service games played
    """
    if pd.isna(sv_gms) or sv_gms == 0:
        return np.nan
    broken = 0
    if pd.notna(bp_faced) and pd.notna(bp_saved):
        broken = max(0, bp_faced - bp_saved)
    return max(0, (sv_gms - broken)) / sv_gms


# Import pandas for compute helpers
import pandas as pd


# ── Quick self-test ──────────────────────────────────────────────────────────

def self_test():
    """Validate the simulator against known results."""
    print("TennisSimulator Self-Test")
    print("=" * 50)

    # Test 1: Equal players (p=0.5 each)
    sim = TennisSimulator(0.5, 0.5)
    print(f"\nEqual players (p=0.5):")
    print(f"  Game hold prob: {sim.g_A:.4f} (should be ~0.5)")
    print(f"  Expected: g=0.5, set should be 50/50")

    r = sim.simulate_match(best_of=3)
    print(f"  P(A wins): {r['p_A_wins_match']:.4f}")
    print(f"  Expected total: {r['expected_total_games']:.1f}")

    # Test 2: Strong server (p=0.70) vs average (p=0.62)
    sim2 = TennisSimulator(0.70, 0.62)
    print(f"\nStrong server (0.70) vs avg (0.62):")
    print(f"  g_A (strong): {sim2.g_A:.4f}")
    print(f"  g_B (avg): {sim2.g_B:.4f}")
    r2 = sim2.simulate_match(best_of=3)
    print(f"  P(A wins): {r2['p_A_wins_match']:.4f}")
    print(f"  Expected total: {r2['expected_total_games']:.1f}")
    print(f"  P(over 22.5): {r2['p_over'][22.5]:.4f}")

    # Test 3: Both big servers (Isner-like) → should produce more games
    sim3 = TennisSimulator(0.73, 0.71)
    print(f"\nTwo big servers (0.73 vs 0.71):")
    print(f"  g_A: {sim3.g_A:.4f}")
    print(f"  g_B: {sim3.g_B:.4f}")
    r3 = sim3.simulate_match(best_of=3)
    print(f"  Expected total: {r3['expected_total_games']:.1f}")
    print(f"  P(over 22.5): {r3['p_over'][22.5]:.4f}")

    # Test 4: BO5 comparison
    r4 = sim2.simulate_match(best_of=5)
    print(f"\nBO5 (strong 0.70 vs avg 0.62):")
    print(f"  Expected total: {r4['expected_total_games']:.1f}")
    print(f"  P(over 38.5): {r4['p_over'][38.5]:.4f}")

    # Test 5: Validate with Monte Carlo
    print(f"\nValidation: Analytical vs Monte Carlo (100k sims)")
    sim5 = TennisSimulator(0.65, 0.62)
    r_anal = sim5.simulate_match(best_of=3)
    r_mc = sim5.simulate_match(best_of=3, n_simulations=100000)
    print(f"  Analytical: expected={r_anal['expected_total_games']:.2f}  P(A)={r_anal['p_A_wins_match']:.4f}  P(>22.5)={r_anal['p_over'][22.5]:.4f}")
    print(f"  MC (100k):  expected={r_mc['expected_total_games']:.2f}  P(A)={r_mc['p_A_wins_match']:.4f}  P(>22.5)={r_mc['p_over'][22.5]:.4f}")


if __name__ == "__main__":
    self_test()
