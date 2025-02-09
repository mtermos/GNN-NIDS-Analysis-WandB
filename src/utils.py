import itertools

import matplotlib.pyplot as plt
import numpy as np
import json
import timeit

from functools import wraps


def calculate_fpr_fnr_with_global(cm):
    """
    Calculate FPR and FNR for each class and globally for a multi-class confusion matrix.

    Parameters:
        cm (numpy.ndarray): Confusion matrix of shape (num_classes, num_classes).

    Returns:
        dict: A dictionary containing per-class and global FPR and FNR.
    """
    num_classes = cm.shape[0]
    results = {"per_class": {}, "global": {}}

    # Initialize variables for global calculation
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0

    # Per-class calculation
    for class_idx in range(num_classes):
        TP = cm[class_idx, class_idx]
        FN = np.sum(cm[class_idx, :]) - TP
        FP = np.sum(cm[:, class_idx]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        # Calculate FPR and FNR for this class
        FPR = FP / (FP + TN) if (FP + TN) != 0 else None
        FNR = FN / (TP + FN) if (TP + FN) != 0 else None

        # Store per-class results
        results["per_class"][class_idx] = {"FPR": FPR, "FNR": FNR}

        # Update global counts
        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

    # Global calculation
    global_FPR = total_FP / \
        (total_FP + total_TN) if (total_FP + total_TN) != 0 else None
    global_FNR = total_FN / \
        (total_FN + total_TP) if (total_FN + total_TP) != 0 else None

    results["global"]["FPR"] = global_FPR
    results["global"]["FNR"] = global_FNR

    return results


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalized=False,
                          file_path=None,
                          show_figure=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalized:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    if file_path:
        plt.savefig(file_path)
    if show_figure:
        plt.show()
    return fig


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if verbose is in kwargs, defaulting to False if not provided
        verbose = kwargs.get("verbose", False)
        if verbose:
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            print(
                f"==>> {func.__name__}: {result}, in {str(timeit.default_timer() - start_time)} seconds")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper
