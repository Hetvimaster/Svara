"""
detection/dual_window_ewma.py

Fast + slow EWMA smoothing over AASIST's own natural sliding windows
(each already NB_SAMP=64600 samples — the exact size the model expects).

Earlier version built an independent 1s "short" window, which
pad_or_trim then looped 4x to fill AASIST's required input length —
feeding the model an unnatural repeated-audio pattern it was never
trained on, which suppressed every score. This version reuses the
window list predict_windowed already computes correctly, and only
changes how those scores get combined ACROSS TIME.
"""

import numpy as np


class DualWindowEWMA:
    def __init__(self, alpha_fast=0.6, alpha_slow=0.2, combine="max"):
        """
        alpha_fast: reacts quickly to a sudden spike (the "short window" intent)
        alpha_slow: only rises on a SUSTAINED pattern (the "long window" intent)
        combine: "max"  -> flag if EITHER trace looks risky (matches your
                 previous, well-working max-aggregate sensitivity)
                 "mean" -> average of both traces (smoother, less reactive)
        """
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.combine = combine

    @staticmethod
    def _ewma(scores, alpha):
        if not scores:
            return []
        trace = [scores[0]]
        for s in scores[1:]:
            trace.append(alpha * s + (1 - alpha) * trace[-1])
        return trace

    def score_from_windows(self, window_scores):
        """window_scores: the list detector.predict_windowed already
        returns as its second value — full 64,600-sample windows,
        never shortened."""
        if not window_scores:
            return {"final_score": 0.0, "fast_trace": [], "slow_trace": []}

        fast_trace = self._ewma(window_scores, self.alpha_fast)
        slow_trace = self._ewma(window_scores, self.alpha_slow)

        # Use the PEAK of each smoothed trace, not just its last value —
        # otherwise a genuine spike mid-clip that decays by the final
        # window gets silently erased.
        fast_peak = max(fast_trace)
        slow_peak = max(slow_trace)

        final_score = max(fast_peak, slow_peak) if self.combine == "max" \
            else (fast_peak + slow_peak) / 2

        return {
            "fast_trace": fast_trace,
            "slow_trace": slow_trace,
            "fast_peak": fast_peak,
            "slow_peak": slow_peak,
            "final_score": final_score,
        }