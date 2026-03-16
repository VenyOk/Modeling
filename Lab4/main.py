import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RANDOM_SEED = 42


def load_samples(file_path):
    data = pd.read_csv(file_path, usecols=["Glucose", "Outcome"])
    data = data[data["Glucose"] > 0].copy()
    ksi0 = data.loc[data["Outcome"] == 0, "Glucose"].to_numpy(dtype=float)
    ksi1 = data.loc[data["Outcome"] == 1, "Glucose"].to_numpy(dtype=float)
    return ksi0, ksi1


def split_sample(sample, train_ratio=0.8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(len(sample))
    np.random.shuffle(indices)
    split_point = int(len(sample) * train_ratio)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    return sample[train_indices], sample[test_indices]


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
    mean_x = np.mean(lnx)
    mean_y = np.mean(lny)
    var_x = np.var(lnx, ddof=1)
    var_y = np.var(lny, ddof=1)
    beta = np.sqrt(var_y / var_x)
    alpha = np.exp(mean_y - beta * mean_x)
    return alpha, beta, mean_y, mean_x, var_y, var_x


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
    ax.set_title(f"Функции распределения Glukose")
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
    ax.set_title(f"После преобразования phi(x)=alpha*x^beta")
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


def print_train_test_results(alpha, beta, d_raw_train, z_raw_train, d_model_train, z_model_train,
                              d_raw_test, z_raw_test, d_model_test, z_model_test):
    print(f"alpha = {alpha:.6f}")
    print(f"beta = {beta:.6f}")
    print(f"\nРабочая выборка:")
    print(f"  max|F(ksi0) - F(ksi1)| = {d_raw_train:.6f}, z = {z_raw_train:.6f}")
    print(f"  max|F(phi(ksi0)) - F(ksi1)| = {d_model_train:.6f}, z = {z_model_train:.6f}")
    if d_model_train > 0:
        improvement_train = d_raw_train / d_model_train
        print(f"  Улучшение: {improvement_train:.2f}x")
    print(f"\nКонтрольная выборка:")
    print(f"  max|F(ksi0) - F(ksi1)| = {d_raw_test:.6f}, z = {z_raw_test:.6f}")
    print(f"  max|F(phi(ksi0)) - F(ksi1)| = {d_model_test:.6f}, z = {z_model_test:.6f}")
    if d_model_test > 0:
        improvement_test = d_raw_test / d_model_test
        print(f"  Улучшение: {improvement_test:.2f}x")


def test_model(ksi0, ksi1, train_ratio, seed=None):
    ksi0_train, ksi0_test = split_sample(ksi0, train_ratio, seed)
    ksi1_train, ksi1_test = split_sample(ksi1, train_ratio, seed)
    
    alpha, beta, _, _, _, _ = estimate_alpha_beta(ksi0_train, ksi1_train)
    
    d_raw_train, z_raw_train = max_distribution_difference(ksi0_train, ksi1_train)
    d_model_train, z_model_train = max_distribution_difference(phi(ksi0_train, alpha, beta), ksi1_train)
    
    d_raw_test, z_raw_test = max_distribution_difference(ksi0_test, ksi1_test)
    d_model_test, z_model_test = max_distribution_difference(phi(ksi0_test, alpha, beta), ksi1_test)
    
    return alpha, beta, d_raw_train, z_raw_train, d_model_train, z_model_train, d_raw_test, z_raw_test, d_model_test, z_model_test


def main():
    ksi0, ksi1 = load_samples("diabetes.csv")
    
    print("=== Разделение 80 (рабочая) / 20 (контрольная) ===")
    alpha_80, beta_80, d_raw_train_80, z_raw_train_80, d_model_train_80, z_model_train_80, d_raw_test_80, z_raw_test_80, d_model_test_80, z_model_test_80 = test_model(ksi0, ksi1, 0.8, RANDOM_SEED)
    print_train_test_results(alpha_80, beta_80, d_raw_train_80, z_raw_train_80, d_model_train_80, z_model_train_80, d_raw_test_80, z_raw_test_80, d_model_test_80, z_model_test_80)
    
    print("\n=== Разделение 70 (рабочая) / 30 (контрольная) ===")
    alpha_70, beta_70, d_raw_train_70, z_raw_train_70, d_model_train_70, z_model_train_70, d_raw_test_70, z_raw_test_70, d_model_test_70, z_model_test_70 = test_model(ksi0, ksi1, 0.7, RANDOM_SEED)
    print_train_test_results(alpha_70, beta_70, d_raw_train_70, z_raw_train_70, d_model_train_70, z_model_train_70, d_raw_test_70, z_raw_test_70, d_model_test_70, z_model_test_70)


if __name__ == "__main__":
    main()
