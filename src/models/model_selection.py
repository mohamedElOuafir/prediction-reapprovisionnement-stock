import numpy as np



def select_best_regression_model(
        results: dict,
        weights: dict = None,
        underestimation_penalty: float = 1.5,
        verbose: bool = True
    ):
    
    if weights is None:
        weights = {"r2": 0.4, "wape": 0.4, "bias": 0.2}
 
    names = list(results.keys())
    summary = {
        n: {
            "r2_mean": np.mean(results[n]["r2"]),
            "wape_mean": np.mean(results[n]["wape"]),
            "bias_mean": np.mean(results[n]["bias"]),
            "r2_std": np.std(results[n]["r2"]),
            "wape_std": np.std(results[n]["wape"]),
        }
        for n in names
    }
 
    def bias_penalty(b):
        return abs(b) if b >= 0 else abs(b) * underestimation_penalty
 
    r2_rank = _rank(names, key=lambda n: -summary[n]["r2_mean"])
    wape_rank = _rank(names, key=lambda n: summary[n]["wape_mean"])
    bias_rank = _rank(names, key=lambda n: bias_penalty(summary[n]["bias_mean"]))
 
    scores = {
        n: weights["r2"] * r2_rank[n]
           + weights["wape"] * wape_rank[n]
           + weights["bias"] * bias_rank[n]
        for n in names
    }
 
    best_name = min(scores, key=scores.get)
 
    if verbose:
        print("=== Classement des modeles de regression ===")
        for n in sorted(scores, key=scores.get):
            s = summary[n]
            print(f"{n:20s} score={scores[n]:.2f} | "
                  f"R2={s['r2_mean']:.3f}(+/-{s['r2_std']:.3f}) | "
                  f"WAPE={s['wape_mean']:.1f}%(+/-{s['wape_std']:.1f}) | "
                  f"Bias={s['bias_mean']:+.2f}%")
        print(f"\n>>> Modele retenu : {best_name}\n")
 
    return best_name, summary, scores


def _rank(names, key):
    ordered = sorted(names, key=key)
    return {n: i + 1 for i, n in enumerate(ordered)}