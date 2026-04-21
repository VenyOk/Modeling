import csv
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import ks_2samp, t as student_t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

DIR = Path(__file__).resolve().parent
SPEARMAN_SIGNIFICANCE_ALPHA = 0.05
EXPORT_RANK_LEVELS = 16
TRAIN_TEST_STRATIFY_BINS = 16
TEST_SET_FRACTION = 0.2
TRAIN_TEST_SPLIT_SEED = 67
TRAIN_TEST_BALANCE_TRIES = 96
KS_SIGNIFICANCE_LEVEL = 0.05
GLUCOSE_EXCLUDE = frozenset({"Glucose", "Outcome", "Insulin"})
PCA_COMPONENTS_GLUCOSE = 4
ZERO_TO_MEDIAN_EXCLUDE = frozenset({"Outcome", "Pregnancies", "Glucose", "Insulin"})


def _replace_zeros_with_median(column, name):
    if name in ZERO_TO_MEDIAN_EXCLUDE:
        return column
    col = np.asarray(column, dtype=float)
    nonzero = col[col != 0]
    if nonzero.size == 0:
        return col
    med = float(np.median(nonzero))
    out = col.copy()
    out[out == 0] = med
    return out


def load_glucose_regression(path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    fieldnames = reader.fieldnames
    pred_names = [name for name in fieldnames if name not in GLUCOSE_EXCLUDE]
    n_pred_expected = len(fieldnames) - len(GLUCOSE_EXCLUDE)
    if len(pred_names) != n_pred_expected:
        raise ValueError(f"ожидается {n_pred_expected} предикторов, получено {len(pred_names)}")
    X = np.array([[float(row[name]) for name in pred_names] for row in rows], dtype=float)
    for j, name in enumerate(pred_names):
        X[:, j] = _replace_zeros_with_median(X[:, j], name)
    y = np.array([float(row["Glucose"]) for row in rows], dtype=float)
    mask = y > 0
    X = X[mask]
    y = y[mask]
    return pred_names, X, y


def fit_standardizer(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    return (X - mean) / std


def _gaussian_solve(A, b):
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy().ravel()
    n = A.shape[0]
    if A.shape[1] != n or b.shape[0] != n:
        raise ValueError("ожидается квадратная A и b той же размерности")
    for k in range(n):
        pivot = int(k + np.argmax(np.abs(A[k:, k])))
        if abs(A[pivot, k]) < 1e-15:
            raise ValueError("матрица нормальных уравнений вырождена или плохо обусловлена")
        if pivot != k:
            A[[k, pivot], :] = A[[pivot, k], :]
            b[[k, pivot]] = b[[pivot, k]]
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        s = b[i] - float(np.dot(A[i, i + 1 :], x[i + 1 :]))
        den = A[i, i]
        if abs(den) < 1e-15:
            raise ValueError("нулевой диагональный элемент при обратной подстановке")
        x[i] = s / den
    return x


def ols_fit(X, y):
    n = X.shape[0]
    X1 = np.column_stack([np.ones(n), X])
    XtX = X1.T @ X1
    Xty = X1.T @ y
    return _gaussian_solve(XtX, Xty)


def ols_predict(X, w):
    X1 = np.column_stack([np.ones(len(X)), X])
    return X1 @ w


def pca_top_components(X, k):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k]


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def metrics_reg(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err_rmse = rmse(y_true, y_pred)
    err_mae = mae(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return err_rmse, err_mae, r2


def rankdata_average(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    sort_order = np.argsort(values, kind="mergesort")
    sorted_vals = values[sort_order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sort_order[k]] = avg
        i = j
    return ranks


def fixed_level_ranks_equal_frequency(values, num_levels):
    values = np.asarray(values, dtype=float)
    n = len(values)
    num_levels = int(num_levels)
    if num_levels < 2:
        raise ValueError("num_levels must be >= 2")
    num_levels = min(num_levels, n)
    order = np.argsort(values, kind="mergesort")
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(n)
    bin_id = (pos * num_levels) // n
    bin_id = np.clip(bin_id, 0, num_levels - 1)
    return (bin_id + 1).astype(float)


def spearman_correlation_matrix(data, num_rank_levels=None):
    n_samples, n_features = data.shape
    if num_rank_levels is None:
        cols = [rankdata_average(data[:, j]) for j in range(n_features)]
    else:
        k = max(2, min(int(num_rank_levels), n_samples))
        cols = [fixed_level_ranks_equal_frequency(data[:, j], k) for j in range(n_features)]
    rank_matrix = np.column_stack(cols)
    coeff = np.corrcoef(rank_matrix, rowvar=False)
    coeff = np.nan_to_num(coeff, nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(coeff, 1.0)
    return coeff


def strength_matrix(matrix):
    strength = np.abs(matrix).astype(float)
    np.fill_diagonal(strength, 0.0)
    return strength


def spearman_criterion_n(n_samples):
    return int(n_samples)


def spearman_pair_criteria(rho, n, alpha):
    if n < 3:
        return None
    k_df = n - 2
    rho = float(np.clip(rho, -1.0, 1.0))
    t_kp = float(student_t.ppf(1.0 - alpha / 2.0, k_df))
    T_kp = t_kp * math.sqrt(max(0.0, 1.0 - rho * rho) / k_df)
    abs_rho = abs(rho)
    return k_df, t_kp, T_kp, abs_rho


def spearman_pair_significant(rho, n, alpha):
    c = spearman_pair_criteria(rho, n, alpha)
    if c is None:
        return False
    k_df, t_kp, T_kp, abs_rho = c
    return abs_rho >= T_kp


def spearman_significant_mask(coeff, n_spearman, alpha):
    p = coeff.shape[0]
    mask = np.zeros((p, p), dtype=bool)
    for i in range(p):
        for j in range(i + 1, p):
            if spearman_pair_significant(coeff[i, j], n_spearman, alpha):
                mask[i, j] = True
                mask[j, i] = True
    return mask


def ranks_work_control_pooled(X_tr, X_te, col):
    v = np.concatenate([X_tr[:, col], X_te[:, col]])
    r = rankdata_average(v)
    n_tr = X_tr.shape[0]
    return r[:n_tr], r[n_tr:]


def kolmogorov_smirnov_pair_stats(x, y, alpha):
    r = ks_2samp(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    d = float(r.statistic)
    p = float(r.pvalue)
    return d, p, p < alpha


def connected_components(strength, significant_mask):
    count = strength.shape[0]
    adjacency = [[] for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            if significant_mask[i, j]:
                adjacency[i].append(j)
                adjacency[j].append(i)
    visited = [False] * count
    components = []
    for start in range(count):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def select_features(strength, significant_mask, feature_names):
    components = connected_components(strength, significant_mask)
    selected = []
    for component in components:
        if len(component) == 1:
            selected.append(component[0])
            continue
        weights = []
        for idx in component:
            score = float(sum(strength[idx, other] for other in component if other != idx))
            weights.append((idx, score))
        representative = sorted(weights, key=lambda item: (-item[1], feature_names[item[0]]))[0][0]
        selected.append(representative)
    global_scores = strength.sum(axis=1)
    if len(selected) < 2:
        order = sorted(range(len(feature_names)), key=lambda idx: (-global_scores[idx], feature_names[idx]))
        for idx in order:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= 2:
                break
    return sorted(set(selected))


def _y_strata_quantile_ranks(y, n_strata_requested):
    y = np.asarray(y, dtype=float)
    n = int(y.shape[0])
    n_strata = max(3, min(int(n_strata_requested), max(3, n // max(4, int(np.ceil(1.0 / TEST_SET_FRACTION))))))
    n_strata = min(n_strata, n // 2)
    r = rankdata_average(y)
    tmp = (r - 1.0) * float(n_strata) / float(max(n, 1))
    sid = np.minimum(np.floor(tmp).astype(np.int64), n_strata - 1)
    sid = np.clip(sid, 0, n_strata - 1).astype(int)
    return sid, n_strata


def covariate_mean_balance_score(X, tr, te):
    X = np.asarray(X, dtype=float)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    d = np.abs(X[tr].mean(axis=0) - X[te].mean(axis=0)) / std
    return float(np.mean(d))


def split_train_test_stratified_y(y, test_fraction, n_strata, seed):
    y = np.asarray(y, dtype=float)
    n = int(y.shape[0])
    rng = np.random.default_rng(seed)
    sid, n_strata = _y_strata_quantile_ranks(y, n_strata)
    buckets = []
    for h in range(n_strata):
        ix = np.where(sid == h)[0]
        if len(ix) == 0:
            continue
        ix = ix.copy()
        rng.shuffle(ix)
        buckets.append(ix)
    if not buckets:
        raise ValueError("empty stratification")
    sizes = np.array([len(b) for b in buckets], dtype=int)
    n_te_target = int(round(n * float(test_fraction)))
    n_te_target = max(1, min(n_te_target, n - 1))
    caps = np.maximum(0, sizes - 1)
    ideal = sizes.astype(np.float64) / n * n_te_target
    n_te = np.minimum(np.floor(ideal).astype(int), caps)
    deficit = n_te_target - int(n_te.sum())
    frac = ideal - np.floor(ideal)
    prio = np.argsort(-frac, kind="mergesort")
    steps = 0
    h_max = len(buckets)
    while deficit > 0 and steps < h_max * n:
        h = int(prio[steps % h_max])
        if n_te[h] < caps[h]:
            n_te[h] += 1
            deficit -= 1
        steps += 1
    if deficit > 0:
        perm = rng.permutation(n)
        n_te_alt = max(1, min(n_te_target, n - 1))
        return perm[n_te_alt:], perm[:n_te_alt]
    tr_list, te_list = [], []
    for i, b in enumerate(buckets):
        nt = int(n_te[i])
        te_list.append(b[:nt])
        tr_list.append(b[nt:])
    return np.concatenate(tr_list), np.concatenate(te_list)


def pick_train_test_stratified_balanced(X, y, test_fraction, n_strata_requested, base_seed, n_tries):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    best_tr, best_te = None, None
    best_score = float("inf")
    for t in range(int(n_tries)):
        seed = int(base_seed) * 1_000_003 + t
        tr, te = split_train_test_stratified_y(y, test_fraction, n_strata_requested, seed)
        score = covariate_mean_balance_score(X, tr, te)
        if score < best_score - 1e-15:
            best_score = score
            best_tr, best_te = tr, te
    return best_tr, best_te, best_score


def save_glucose_regression_diagnostics(path, y_tr, y_te, models):
    y_tr = np.asarray(y_tr, dtype=float)
    y_te = np.asarray(y_te, dtype=float)
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 3.5 * n_models))
    if n_models == 1:
        axes = np.reshape(axes, (1, -1))
    lo = float(min(y_tr.min(), y_te.min()))
    hi = float(max(y_tr.max(), y_te.max()))
    pad = 0.03 * (hi - lo + 1e-9)
    lim_lo, lim_hi = lo - pad, hi + pad
    for row, (label, ptr, pte) in enumerate(models):
        ptr = np.asarray(ptr, dtype=float)
        pte = np.asarray(pte, dtype=float)
        ax_sc, ax_r = axes[row, 0], axes[row, 1]
        ax_sc.scatter(y_tr, ptr, alpha=0.4, s=14, c="C0", label="train")
        ax_sc.scatter(y_te, pte, alpha=0.55, s=18, c="C1", marker="^", label="test", edgecolors="none")
        ax_sc.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.0, label="y = x")
        ax_sc.set_xlim(lim_lo, lim_hi)
        ax_sc.set_ylim(lim_lo, lim_hi)
        ax_sc.set_aspect("equal")
        ax_sc.set_xlabel("Glucose, факт")
        ax_sc.set_ylabel("Glucose, предсказание")
        ax_sc.set_title(label)
        ax_sc.legend(loc="upper left", fontsize=8)
        ax_sc.grid(True, alpha=0.3)
        res_tr = y_tr - ptr
        res_te = y_te - pte
        ax_r.scatter(ptr, res_tr, alpha=0.4, s=14, c="C0", label="train")
        ax_r.scatter(pte, res_te, alpha=0.55, s=18, c="C1", marker="^", label="test", edgecolors="none")
        ax_r.axhline(0.0, color="k", lw=1.0)
        ax_r.set_xlabel("Предсказание")
        ax_r.set_ylabel("Остаток (факт - предсказание)")
        ax_r.set_title(f"{label}: остатки")
        ax_r.legend(loc="upper right", fontsize=8)
        ax_r.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    data_csv = DIR / "diabetes.csv"
    if not data_csv.is_file():
        raise FileNotFoundError(f"Ожидается файл данных: {data_csv}")

    pred_names, X, y = load_glucose_regression(data_csv)
    ols_full_tag = f"ols_{len(pred_names)}"
    n_obs = len(y)
    coeff_6 = spearman_correlation_matrix(X, EXPORT_RANK_LEVELS)
    strength_6 = strength_matrix(coeff_6)
    n_sp = spearman_criterion_n(n_obs)
    sig_mask = spearman_significant_mask(coeff_6, n_sp, SPEARMAN_SIGNIFICANCE_ALPHA)
    selected_indices = select_features(strength_6, sig_mask, pred_names)
    selected_names = [pred_names[i] for i in selected_indices]
    print("Признаки после отбора Spearman:", ", ".join(selected_names))
    tr, te, balance_score = pick_train_test_stratified_balanced(
        X,
        y,
        TEST_SET_FRACTION,
        TRAIN_TEST_STRATIFY_BINS,
        TRAIN_TEST_SPLIT_SEED,
        TRAIN_TEST_BALANCE_TRIES,
    )
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    n_tr = int(X_tr.shape[0])
    n_te = int(X_te.shape[0])
    mean, std = fit_standardizer(X_tr)
    X_tr_s = apply_standardizer(X_tr, mean, std)
    X_te_s = apply_standardizer(X_te, mean, std)
    w6 = ols_fit(X_tr_s, y_tr)
    pred_tr_6 = ols_predict(X_tr_s, w6)
    pred_te_6 = ols_predict(X_te_s, w6)
    split_rmse, split_mae, split_r2 = metrics_reg(y_te, pred_te_6)
    print(
        f"Разбиение train/test: страты по рангам Glucose, баланс средних X (норм.)={balance_score:.6f}, "
        f"кандидатов={TRAIN_TEST_BALANCE_TRIES}, seed_база={TRAIN_TEST_SPLIT_SEED}, доля test={TEST_SET_FRACTION}; "
        f"полная OLS на test: R^2={split_r2:.6f}, RMSE={split_rmse:.6f}, MAE={split_mae:.6f}"
    )
    X_tr_sel = X_tr[:, selected_indices]
    X_te_sel = X_te[:, selected_indices]
    mean_sel, std_sel = fit_standardizer(X_tr_sel)
    X_tr_ss = apply_standardizer(X_tr_sel, mean_sel, std_sel)
    X_te_ss = apply_standardizer(X_te_sel, mean_sel, std_sel)
    w_sel = ols_fit(X_tr_ss, y_tr)
    pred_tr_sel = ols_predict(X_tr_ss, w_sel)
    pred_te_sel = ols_predict(X_te_ss, w_sel)
    n_sel = len(selected_indices)
    k_pca = min(PCA_COMPONENTS_GLUCOSE, n_sel)
    Vk = pca_top_components(X_tr_ss, k_pca)
    Z_tr = X_tr_ss @ Vk.T
    Z_te = X_te_ss @ Vk.T
    w_pca = ols_fit(Z_tr, y_tr)
    pred_tr_pca = ols_predict(Z_tr, w_pca)
    pred_te_pca = ols_predict(Z_te, w_pca)
    m_tr_6 = metrics_reg(y_tr, pred_tr_6)
    m_te_6 = metrics_reg(y_te, pred_te_6)
    m_tr_sel = metrics_reg(y_tr, pred_tr_sel)
    m_te_sel = metrics_reg(y_te, pred_te_sel)
    m_tr_pca = metrics_reg(y_tr, pred_tr_pca)
    m_te_pca = metrics_reg(y_te, pred_te_pca)
    reg_plot_path = DIR / "glucose_regression_diagnostics.png"
    save_glucose_regression_diagnostics(
        reg_plot_path,
        y_tr,
        y_te,
        [
            (ols_full_tag, pred_tr_6, pred_te_6),
            ("OLS (отбор Spearman)", pred_tr_sel, pred_te_sel),
            (f"OLS + PCA{k_pca}", pred_tr_pca, pred_te_pca),
        ],
    )
    print(f"Графики регрессии Glucose: {reg_plot_path}")
    print(
        "Метрики RMSE, MAE и R^2:\n"
        f"  {ols_full_tag}:               train RMSE={m_tr_6[0]:.6f}, MAE={m_tr_6[1]:.6f}, R^2={m_tr_6[2]:.6f} | "
        f"test RMSE={m_te_6[0]:.6f}, MAE={m_te_6[1]:.6f}, R^2={m_te_6[2]:.6f}\n"
        f"  ols_spearman_selected: train RMSE={m_tr_sel[0]:.6f}, MAE={m_tr_sel[1]:.6f}, R^2={m_tr_sel[2]:.6f} | "
        f"test RMSE={m_te_sel[0]:.6f}, MAE={m_te_sel[1]:.6f}, R^2={m_te_sel[2]:.6f}\n"
        f"  ols_pca{k_pca}:          train RMSE={m_tr_pca[0]:.6f}, MAE={m_tr_pca[1]:.6f}, R^2={m_tr_pca[2]:.6f} | "
        f"test RMSE={m_te_pca[0]:.6f}, MAE={m_te_pca[1]:.6f}, R^2={m_te_pca[2]:.6f}"
    )
    p_feat = len(pred_names)
    for i in range(p_feat):
        for j in range(i + 1, p_feat):
            rho_sp = float(coeff_6[i, j])
            _, r_te_i = ranks_work_control_pooled(X_tr, X_te, i)
            _, r_te_j = ranks_work_control_pooled(X_tr, X_te, j)
            _, _, ks_ok = kolmogorov_smirnov_pair_stats(r_te_i, r_te_j, KS_SIGNIFICANCE_LEVEL)
            sp_c = spearman_pair_criteria(rho_sp, n_sp, SPEARMAN_SIGNIFICANCE_ALPHA)
            ks_status = "подтвердилась" if ks_ok else "не подтвердилась"
            if sp_c is None:
                ok_val = "н/д"
            else:
                _, _, T_kp, abs_rho = sp_c
                sp_ok = abs_rho >= T_kp
                ok_val = "да" if sp_ok else "нет"
            print(f"{pred_names[i]} — {pred_names[j]}: OK = {ok_val}, КС {ks_status}")


if __name__ == "__main__":
    main()
