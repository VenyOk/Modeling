import numpy as np
import pandas as pd


def load_samples(file_path):
    data = pd.read_csv(file_path, usecols=["Glucose", "Outcome"])
    data = data[data["Glucose"] > 0].copy()
    ksi0 = data.loc[data["Outcome"] == 0, "Glucose"].to_numpy(dtype=float)
    ksi1 = data.loc[data["Outcome"] == 1, "Glucose"].to_numpy(dtype=float)
    return ksi0, ksi1


def max_distribution_difference(sample_1, sample_2):
    sample_1 = np.sort(np.asarray(sample_1, dtype=float))
    sample_2 = np.sort(np.asarray(sample_2, dtype=float))
    z = np.sort(np.unique(np.concatenate([sample_1, sample_2])))
    F1 = np.searchsorted(sample_1, z, side="right") / len(sample_1)
    F2 = np.searchsorted(sample_2, z, side="right") / len(sample_2)
    diff = np.abs(F1 - F2)
    i = np.argmax(diff)
    return diff[i], z[i]


def estimate_alpha_beta(ksi0, ksi1):
    lnx = np.log(ksi0)
    lny = np.log(ksi1)
    A = np.mean(lny)
    B = np.mean(lnx)
    C = np.mean((lny - A) ** 2)
    D = np.mean((lnx - B) ** 2)
    beta = np.sqrt(C / D)
    alpha = np.exp(A - beta * B)
    return float(alpha), float(beta)


def phi(x, alpha, beta):
    return alpha * x**beta


def trim_tails(sample, low_q, high_q):
    sample = np.asarray(sample, dtype=float)
    if low_q <= 0 and high_q >= 1:
        return sample
    lo = np.quantile(sample, low_q)
    hi = np.quantile(sample, high_q)
    return sample[(sample >= lo) & (sample <= hi)]


def run_for_q(ksi0, ksi1, q):
    x0 = trim_tails(ksi0, q, 1 - q)
    x1 = trim_tails(ksi1, q, 1 - q)
    alpha, beta = estimate_alpha_beta(x0, x1)
    d_raw, _ = max_distribution_difference(x0, x1)
    d_model, _ = max_distribution_difference(phi(x0, alpha, beta), x1)
    return {
        "q": float(q),
        "n0": int(len(x0)),
        "n1": int(len(x1)),
        "alpha": float(alpha),
        "beta": float(beta),
        "d_raw": float(d_raw),
        "d_model": float(d_model),
    }


def main():
    ksi0, ksi1 = load_samples("diabetes.csv")
    qs = [round(x, 4) for x in np.arange(0.0, 0.0301, 0.0005)]
    results = [run_for_q(ksi0, ksi1, q) for q in qs]
    best = min(results, key=lambda r: (r["d_model"], r["q"]))
    for r in results:
        print(
            f"q={r['q']:.3f} n0={r['n0']} n1={r['n1']} "
            f"alpha={r['alpha']:.6f} beta={r['beta']:.6f} d={r['d_model']:.6f}"
        )
    print(f"BEST q={best['q']:.3f} alpha={best['alpha']:.6f} beta={best['beta']:.6f} d={best['d_model']:.6f}")


if __name__ == "__main__":
    main()

