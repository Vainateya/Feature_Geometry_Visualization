from datasets import load_dataset, Value
import torch


def tokenize(dataset, tokenizer, device, max_length=512, add_special_tokens: bool = False) -> list[torch.Tensor]:
    """
    Tokenizes and pads the dataset to fixed max_length.

    Args:
        dataset (list[str]): Input texts (stories or tokens).
        tokenizer: HF tokenizer.
        device: Target device (GPU/CPU).
        max_length (int): Max length per sample.
        add_special_tokens (bool): Whether to automatically add special tokens (eg. BOS, EOS) while tokenizing. Defaults to False

    Returns:
        List[torch.Tensor]: List of (1, seq_len) tokenized tensors.
    """
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = []
    for text in dataset:
        ids = tokenizer.encode(
            text, max_length=max_length, truncation=True, padding="max_length", add_special_tokens=add_special_tokens
        )
        tokenized.append(torch.tensor(ids).unsqueeze(0).to(device))
    return tokenized


def _resolve_dataset_name(dataset: str) -> str:
    """
    Identifies whether the user intended to specify any of the two datasets that LatentLens already supports.
    If tinystories or project gutenberg are mentioned, the correct dataset string is routed and returned.

    Args:
        dataset (str): The specified dataset string given by the user

    Returns:
        str: The correct routed string if matched with supported dataset or the original string
    """
    ds = dataset.strip().lower()
    if ds in {"tinystories", "tiny_stories", "roneneldan/tinystories"}:
        return "roneneldan/TinyStories"
    if ds in {"pg19", "emozilla/pg19"}:
        return "emozilla/pg19"
    return dataset  # assume given HF repo id or local path


def _pick_text_column(ds) -> str:
    """
    Finds the appropriate field in a given dataset which contains the raw text input.
    While most datasets store this information under "text" column, it also checks if
    there exists any columns with the datatype of "string" or "large_string" in an
    attempt to find it.

    Args:
        ds: An appropriate huggingface compatible dataset object fed through load_dataset()

    Returns:
        str: the appropriate field name that contains the raw text (or throws a value error if one wasn't found)
    """
    # Strong preference for the common case
    if "text" in ds.column_names:
        return "text"
    # Fallback: any string/large_string column(s)
    candidates = [c for c, f in ds.features.items() if isinstance(f, Value) and f.dtype in {"string", "large_string"}]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Could not determine a text column. Found string-like columns={candidates} "
        f"among all columns={ds.column_names}. "
        f"Pass a dataset with a 'text' column or add a selector."
    )


def load_dataset_for_gen(dataset: str = "pg19", num_samples: int = 512, split: str = "train") -> list[str]:
    """
    Load exactly `num_samples` documents from a HF dataset (or local HF-compatible path)

    Args:
        dataset (str): The string identifier for this HF, or HF compatible dataset
        num_samples (int): the number of stories, or samples to load
        split (str): the train or test split can be specifed for loading

    Returns:
        list[str]: The list of string text stories from the specified dataset
    """
    name_or_path = _resolve_dataset_name(dataset)
    ds = load_dataset(name_or_path, split=split)

    # Determine the text field robustly
    text_col = _pick_text_column(ds)

    # Pull the first num_samples non-empty stories
    stories = []
    for ex in ds:
        txt = (ex[text_col] or "").strip()
        if txt:
            stories.append(txt)
        if len(stories) >= num_samples:
            break

    if len(stories) < num_samples:
        raise ValueError(
            f"Requested {num_samples} stories, but only found {len(stories)} non-empty in split '{split}' for '{name_or_path}'."
        )
    return stories


def load_single_tokens(tokenizer, device, vocab_size=1000):
    """
    Loads the top-N token IDs directly from tokenizer vocab.

    Args:
        tokenizer: GPT-2 tokenizer.
        device: Torch device.
        vocab_size (int): Number of tokens to extract. If None, use full vocab.

    Returns:
        List[torch.Tensor]: token tensors.
    """
    tokenized = [torch.tensor([[token_id]]).to(device) for token_id in range(vocab_size)]
    return tokenized


def load_repeated_tokens(tokenizer, device, token_id: int, seq_len: int, batch_size: int) -> list[torch.Tensor]:
    """
    Create a batch of sequences where each sequence is length `seq_len`,
    filled with `token_id`, except the first position is BOS if the tokenizer has one.
    Returns a list of tensors shaped (1, seq_len), length=batch_size.

    Args:
        tokenizer: the huggingface (HF) tokenizer object that will be used
        device: the hardware device being used
        token_id(int): the token_id to be repeated
        seq_len(int): the sequence length per sample
        batch_size(int): number of samples

    Returns:
        list[torch.Tensor]: the single token data as a list of tensors
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)
    ids = [token_id] * seq_len
    if bos_id is not None and seq_len > 0:
        ids[0] = bos_id
    one = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    return [one.clone() for _ in range(batch_size)]
