import warnings
import torch
import random

try:
    import pynvml

    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


def select_best_device(mode="m", suppress_output=False):
    """
    Select the best available GPU device based on specified criteria.

    Args:
        mode (str): Selection mode - 'm' for most free memory, 'u' for least utilization. 'u' requires pynvml package to be installed.
        suppress_output (bool): If True, suppress all print and warning outputs

    Returns:
        torch.device: Selected device (GPU or CPU if no GPU available).

    Raises:
        Exception: If mode is not 'm' or 'u'.
    """
    if not torch.cuda.is_available():
        if not suppress_output:
            print("select_best_device(): Using CPU")
        return torch.device("cpu")

    if mode not in ["m", "u"]:
        raise Exception(
            f'select_device_with_most_free_memory: Acceptable inputs for mode are "m" (most free memory) and "u" (least utilization_). You specified: {mode}'
        )

    indices = list(range(torch.cuda.device_count()))
    random.shuffle(
        indices
    )  # shuffle the indices we iterate through so that, if, say, a bunch of processes scramble for GPUs at once, the first one won't get them all

    if mode == "m":
        max_free_memory = 0
        device_index = 0
        for i in indices:
            free_memory = torch.cuda.mem_get_info(i)[0]
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                device_index = i
        return torch.device(f"cuda:{device_index}")

    elif mode == "u":
        if HAS_PYNVML:
            pynvml.nvmlInit()
            min_util = 100
            device_index = 0
            for i in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # Get the handle for the target GPU
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu  # GPU utilization percentage (integer)
                if gpu_utilization < min_util:
                    min_util = gpu_utilization
                    device_index = i
            pynvml.nvmlShutdown()

            # If all the GPUs are basically at max util, then make choice via memory availiability
            if min_util > 95:
                return select_best_device(mode="m")

            return torch.device(f"cuda:{device_index}")
        else:
            if not suppress_output:
                warnings.warn(
                    "Utilization 'u' based selection is only available if pnyvml is available, but it is not. Please install pnyvml to use mode 'u'. Switching to mode 'm' (memory-based device selection)"
                )
            return select_best_device("m")
