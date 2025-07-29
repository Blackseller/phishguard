if url_model:  # Use ML model if available
    prediction = url_model.predict([features])[0]
    return prediction
else:  # Fallback to rule-based scoring
    score = rule_based_score(features)
    return min(100, int(score * 100))
