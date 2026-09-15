def compute_risk_score(spoof_score, speaker_similarity, replay_result):
    """
    spoof_score: 0-1, probability audio is synthetic (from AASIST)
    speaker_similarity: 0-1 cosine similarity, 1 = matches claimed identity
    replay_result: dict from classify_replay(), e.g. {"classification": "live", "confidence": 0.6}

    Returns a 0-100 risk score.
    """
    spoof_risk = spoof_score * 100
    identity_risk = (1 - speaker_similarity) * 100

    # replay/conversion contributes risk only if it's NOT "live",
    # scaled by the heuristic's own confidence
    classification = replay_result["classification"]
    confidence = replay_result["confidence"]
    if classification == "live":
        replay_risk = 0
    else:
        replay_risk = 100 * confidence

    # weighted sum — day-one version, tune weights once you have labeled test data
    risk = (0.4 * spoof_risk) + (0.4 * identity_risk) + (0.2 * replay_risk)
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
    dummy_replay = {"classification": "live", "confidence": 0.6}
    score = compute_risk_score(spoof_score=0.1, speaker_similarity=0.95, replay_result=dummy_replay)
    print(f"Risk score: {score:.2f}, Tier: {get_tier(score)}")