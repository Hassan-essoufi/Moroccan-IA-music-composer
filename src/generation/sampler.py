import numpy as np

def softmax(logits):
    """
    Compute softmax probabilities from logits.
    """
    logits = logits - np.max(logits)  # numerical stability
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def apply_temperature(logits, temperature):
    """
    Apply temperature scaling to logits.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    return logits / temperature


def top_k_sampling(logits, k):
    """
    top_k sampling.
    """
    if k <= 0 or k >= len(logits):
        return logits

    indices = np.argpartition(logits, -k)[-k:]
    mask = np.full_like(logits, -np.inf)
    mask[indices] = logits[indices]
    return mask


def top_p_sampling(logits, p):
    """
    Nucleus (top-p) sampling.
    """
    if p <= 0 or p > 1:
        raise ValueError("top_p must be in (0, 1]")

    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = cumulative_probs <= p
    cutoff[0] = True  # always keep at least one token

    selected_indices = sorted_indices[cutoff]

    mask = np.full_like(logits, -np.inf)
    mask[selected_indices] = logits[selected_indices]
    return mask


def sample_next_token(
    logits,
    temperature=1.0,
    top_k=None,
    top_p=None
):
    """
    Sample next token from logits using temperature, top-k and/or top-p.
    """
    logits = np.asarray(logits).astype(np.float64)

    # temperature
    logits = apply_temperature(logits, temperature)

    # top-k
    if top_k is not None:
        logits = top_k_sampling(logits, top_k)

    # top-p
    if top_p is not None:
        logits = top_p_sampling(logits, top_p)

    probs = softmax(logits)

    if np.any(np.isnan(probs)) or np.sum(probs) == 0:
        raise ValueError("invalid probability distribution during sampling")

    return np.random.choice(len(probs), p=probs)
