import random


def decide(probability: float) -> bool:
    """
    Randomly make a binary decision given the probability of decision outcome

    Args:
        probability: That a customer will consume

    Returns: Customer's decision

    """

    if probability < 0 or probability > 1:
        raise ValueError(
            "Probability must be a float between 0 and 1, inclusive. Value received: %.2f"
            % probability
        )
    return random.random() < probability


# def consume():
