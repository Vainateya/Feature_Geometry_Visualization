import random 

def get_permutation(strategy: str, num_layers: int) -> list[int]:
    """
    Generate a permutation of layer indices based on a specified strategy

    Parameters:
        strategy (str): The permutation strategy to use. Supported values are:
            - "random": Shuffle the blocks randomly
            - "reverse": Reverse the order of blocks
            - "identity": Return the original order of layer indices (default)
            - "swap:i,j": Swap the associated blocks (e.g., "swap:1,3").
            - "custom:i1,i2,...,in": Use a custom permutation specified explicitly (e.g., "custom:2,0,1,3...")
        num_layers (int): The total number of layers (defines the range [0, num_layers))

    Returns:
        list[int]: A list of integers representing the permutation
    """

    if strategy == "random":
        perm = list(range(num_layers))
        random.shuffle(perm)
        return perm
    elif strategy == "reverse":
        return list(reversed(range(num_layers)))
    elif strategy == "identity":
        return list(range(num_layers))
    elif strategy.startswith("swap:"):
        i, j = map(int, strategy.split(":")[1].split(","))
        perm = list(range(num_layers))
        perm[i], perm[j] = perm[j], perm[i]
        return perm
    elif strategy.startswith("custom:"):
        return list(map(int, strategy.split(":")[1].split(",")))
    else:
        raise ValueError(f"Unrecognized permutation strategy: {strategy}")