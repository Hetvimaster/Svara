def compute_risk_score(spoof_score, speaker_similarity):
    """
    spoof_score: 0-1, probability audio is synthetic (from AASIST)
    speaker_similarity: 0-1 cosine similarity, 1 = matches claimed identity

    Returns a 0-100 risk score.
    """
    spoof_risk = spoof_score * 100
    identity_risk = (1 - speaker_similarity) * 100

    # equal weighting — same ratio as the tuned 3-signal version (0.4:0.4),
    # renormalized to sum to 1.0 now that replay_risk is removed
    risk = (0.5 * spoof_risk) + (0.5 * identity_risk)
    return min(100, max(0, risk))

def get_tier(risk_score):
    if risk_score <= 40:
        return "low"
    elif risk_score <= 75:
        return "medium"
    else:
        return "high"

if __name__ == "__main__":
    # sanity check with dummy values
    score = compute_risk_score(spoof_score=0.1, speaker_similarity=0.95)
    print(f"Risk score: {score:.2f}, Tier: {get_tier(score)}")