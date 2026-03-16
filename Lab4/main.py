import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRIM_Q = 0.022


def load_samples(file_path):
    data = pd.read_csv(file_path, usecols=["Glucose", "Outcome"])
    data = data[data["Glucose"] > 0].copy()
    ksi0 = data.loc[data["Outcome"] == 0, "Glucose"].to_numpy(dtype=float)
    ksi1 = data.loc[data["Outcome"] == 1, "Glucose"].to_numpy(dtype=float)
    return ksi0, ksi1


def empirical_distribution(sample):
    sample = np.sort(np.asarray(sample, dtype=float))
    F = np.arange(1, len(sample) + 1) / len(sample)
    return sample, F


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
    return alpha, beta, A, B, C, D


def phi(x, alpha, beta):
    return alpha * x**beta


def trim_tails(sample, low_q, high_q):
    sample = np.asarray(sample, dtype=float)
    if low_q <= 0 and high_q >= 1:
        return sample
    lo = np.quantile(sample, low_q)
    hi = np.quantile(sample, high_q)
    trimmed = sample[(sample >= lo) & (sample <= hi)]
    return trimmed


def save_distribution_plots(ksi0, ksi1, alpha, beta, d_raw, d_model):
    x0, F0 = empirical_distribution(ksi0)
    x1, F1 = empirical_distribution(ksi1)
    phi_x0, F_phi = empirical_distribution(phi(ksi0, alpha, beta))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(x0, F0, where="post", label="F(ksi0)")
    ax.step(x1, F1, where="post", label="F(ksi1)")
    ax.set_title(f"Функции распределения Glucose\nmax|F0-F1| = {d_raw:.4f}")
    ax.set_xlabel("z")
    ax.set_ylabel("F(z)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("glucose_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(phi_x0, F_phi, where="post", label="F(phi(ksi0))")
    ax.step(x1, F1, where="post", label="F(ksi1)")
    ax.set_title(f"После преобразования phi(x)=alpha*x^beta\nmax|F(phi(ksi0))-F(ksi1)| = {d_model:.4f}")
    ax.set_xlabel("z")
    ax.set_ylabel("F(z)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("glucose_transformed_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_results(alpha, beta, d_raw, z_raw, d_model, z_model):
    print(f"alpha = {alpha:.6f}")
    print(f"beta = {beta:.6f}")
    print(f"max|F(ksi0) - F(ksi1)| = {d_raw:.6f}, z = {z_raw:.6f}")
    print(f"max|F(phi(ksi0)) - F(ksi1)| = {d_model:.6f}, z = {z_model:.6f}")


def main():
    ksi0, ksi1 = load_samples("diabetes.csv")
    ksi0 = trim_tails(ksi0, TRIM_Q, 1 - TRIM_Q)
    ksi1 = trim_tails(ksi1, TRIM_Q, 1 - TRIM_Q)
    alpha, beta, _, _, _, _ = estimate_alpha_beta(ksi0, ksi1)
    d_raw, z_raw = max_distribution_difference(ksi0, ksi1)
    d_model, z_model = max_distribution_difference(phi(ksi0, alpha, beta), ksi1)
    save_distribution_plots(ksi0, ksi1, alpha, beta, d_raw, d_model)
    print_results(alpha, beta, d_raw, z_raw, d_model, z_model)


if __name__ == "__main__":
    main()
