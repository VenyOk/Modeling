import csv
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import ks_2samp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

DIR = Path(__file__).resolve().parent
SPEARMAN_EDGE_THRESHOLD = 0.3
EXPORT_RANK_LEVELS = 16
RANDOM_SEED = 42
LOG_FLOOR = 1e-9
KS_SIGNIFICANCE_LEVEL = 0.05


def load_data(path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    feature_names = [name for name in reader.fieldnames if name != "Outcome"]
    data = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    outcome = np.array([int(row["Outcome"]) for row in rows], dtype=int)
    return feature_names, data, outcome


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


def save_matrix_csv(feature_names, matrix, path):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["feature"] + feature_names)
        for feature_name, row in zip(feature_names, matrix):
            writer.writerow([feature_name] + [f"{value:.6f}" for value in row])


def draw_spearman_matrix_table(feature_names, matrix, path, title="Матрица корреляции Спирмена (ранги)"):
    size = len(feature_names)
    fig, axis = plt.subplots(figsize=(2.4 * size, 0.9 * size + 2))
    axis.axis("off")
    cell_text = [[f"{value:.4f}" for value in row] for row in matrix]
    table = axis.table(
        cellText=cell_text,
        rowLabels=feature_names,
        colLabels=feature_names,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.5)
    center = 0.0
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.8)
        if row == 0 or col == -1:
            cell.set_facecolor("#d9e6f2")
            cell.set_text_props(weight="bold")
        else:
            value = matrix[row - 1, col]
            delta = value - center
            if row - 1 == col:
                cell.set_facecolor("#fff4b8")
            elif delta >= 0.2:
                cell.set_facecolor("#fde2e2")
            elif delta <= -0.2:
                cell.set_facecolor("#e3f2e1")
            else:
                cell.set_facecolor("#f4f4f4")
    axis.set_title(title, pad=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def strength_matrix(matrix):
    strength = np.abs(matrix).astype(float)
    np.fill_diagonal(strength, 0.0)
    return strength


def connected_components(strength, threshold):
    count = strength.shape[0]
    adjacency = [[] for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            if strength[i, j] >= threshold:
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


def select_features(strength, threshold, feature_names):
    components = connected_components(strength, threshold)
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


"""
def median_binarize(data):
    medians = np.median(data, axis=0)
    return medians, data > medians


def standardize(points):
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    standardized = (points - mean) / std
    return standardized


def kmeans_once(points, k, max_iterations, rng):
    indices = rng.sample(range(len(points)), k)
    centroids = points[np.array(indices, dtype=int)].astype(float)
    labels = None
    for _ in range(max_iterations):
        distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(distances, axis=1)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if np.any(mask):
                centroids[cluster_id] = points[mask].mean(axis=0)
            else:
                centroids[cluster_id] = points[rng.randrange(len(points))]
    inertia = float(((points - centroids[labels]) ** 2).sum())
    return labels.astype(int), centroids.astype(float), inertia


def kmeans(points, k=2, n_init=30, max_iterations=300, seed=42):
    best = None
    for run in range(n_init):
        rng = random.Random(seed + run * 9973)
        labels, centroids, inertia = kmeans_once(points, k, max_iterations, rng)
        if best is None or inertia < best["inertia"]:
            best = {"labels": labels, "centroids": centroids, "inertia": inertia}
    return best


def align_clusters_to_outcome(labels, outcome):
    labels = np.asarray(labels, dtype=int)
    outcome = np.asarray(outcome, dtype=int)
    direct = float((labels == outcome).mean())
    flipped_labels = 1 - labels
    flipped = float((flipped_labels == outcome).mean())
    if flipped > direct:
        return flipped_labels, flipped
    return labels, direct


def pca_projection(points, dimensions=2):
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:dimensions].T
    projection = centered @ components
    return projection, components, points.mean(axis=0)


def draw_cluster_projection(points_standardized, labels_aligned, centroids_standardized, path):
    projection, components, mean = pca_projection(points_standardized, dimensions=2)
    centroid_projection = (centroids_standardized - mean) @ components
    fig, axis = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#d62728"]
    labels_aligned = np.asarray(labels_aligned, dtype=int)
    for cluster_id in [0, 1]:
        cluster_points = projection[labels_aligned == cluster_id]
        axis.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=24,
            alpha=0.75,
            color=colors[cluster_id],
            label=f"Класс {cluster_id}",
        )
    axis.scatter(
        centroid_projection[:, 0],
        centroid_projection[:, 1],
        s=160,
        c="#000000",
        marker="X",
        label="Центроиды",
    )
    axis.set_xlabel("ГК1")
    axis.set_ylabel("ГК2")
    axis.set_title("Кластеры k-means в 2D проекции PCA (после отбора по Спирмену)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_clusters_csv(data, feature_names, selected_indices, labels_raw, labels_aligned, outcome, path):
    selected_names = [feature_names[idx] for idx in selected_indices]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id_объекта"] + selected_names + ["кластер_сырой", "кластер_сопоставленный", "Исход"])
        for i in range(len(data)):
            values = [f"{data[i, idx]:.6f}" for idx in selected_indices]
            writer.writerow([i + 1] + values + [int(labels_raw[i]), int(labels_aligned[i]), int(outcome[i])])


def save_cluster_summary_csv(feature_names, selected_indices, medians, data, labels_aligned, path):
    selected_names = [feature_names[idx] for idx in selected_indices]
    header = ["класс", "количество"]
    for name in selected_names:
        header.append(f"{name}_центроид")
        header.append(f"{name}_относительно_медианы")
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for class_id in [0, 1]:
            mask = labels_aligned == class_id
            class_data = data[mask][:, selected_indices]
            if len(class_data):
                centroid = class_data.mean(axis=0)
            else:
                centroid = np.zeros(len(selected_indices), dtype=float)
            row = [class_id, int(mask.sum())]
            for local_idx, feature_idx in enumerate(selected_indices):
                value = float(centroid[local_idx])
                direction = "выше_медианы" if value > medians[feature_idx] else "ниже_медианы"
                row.extend([f"{value:.6f}", direction])
            writer.writerow(row)


def build_truth_table_rows(binary_data, labels_aligned):
    counts_by_pattern = {}
    for bits, label in zip(binary_data.astype(int), np.asarray(labels_aligned, dtype=int)):
        pattern = tuple(int(bit) for bit in bits)
        if pattern not in counts_by_pattern:
            counts_by_pattern[pattern] = [0, 0]
        counts_by_pattern[pattern][int(label)] += 1
    truth_table_rows = []
    for pattern in sorted(counts_by_pattern):
        count0, count1 = counts_by_pattern[pattern]
        y = 1 if count1 > count0 else 0
        truth_table_rows.append(list(pattern) + [y])
    return truth_table_rows


def save_truth_table_csv(feature_names, truth_table_rows, path):
    variable_names = [f"x{i}" for i in range(1, len(feature_names) + 1)]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(variable_names + ["y"])
        writer.writerows(truth_table_rows)


def count_binary_labels(labels):
    labels = np.asarray(labels, dtype=int)
    return int((labels == 0).sum()), int((labels == 1).sum())


def count_truth_table_labels(truth_table_rows):
    labels = [row[-1] for row in truth_table_rows]
    return count_binary_labels(labels)


def print_text_output(feature_names, selected_indices, truth_table_counts, cluster_counts, selection_note):
    selected = ", ".join(feature_names[idx] for idx in selected_indices)
    print(f"признаки: {selected}")
    print(selection_note)
    print(f"истинность 0/1: {truth_table_counts[0]}/{truth_table_counts[1]}")
    print(f"кластеры 0/1: {cluster_counts[0]}/{cluster_counts[1]}")


def sweep_rank_levels(feature_names, data, outcome):
    p = len(feature_names)
    rows = []
    for num_levels in RANK_LEVELS_FOR_SWEEP:
        coeff = spearman_correlation_matrix(data, num_levels)
        strength = strength_matrix(coeff)
        selected = select_features(strength, SPEARMAN_EDGE_THRESHOLD, feature_names)
        names = ";".join(feature_names[i] for i in selected)
        off = coeff.astype(float).copy()
        np.fill_diagonal(off, 0.0)
        mean_abs = float(np.abs(off).sum() / max(p * (p - 1), 1))
        sel_points = data[:, selected]
        std_pts = standardize(sel_points)
        km = kmeans(std_pts, k=2, n_init=30, max_iterations=300, seed=RANDOM_SEED)
        _, acc = align_clusters_to_outcome(km["labels"], outcome)
        label = "полные" if num_levels is None else str(int(num_levels))
        rows.append(
            {
                "rank_levels": label,
                "selected_count": len(selected),
                "selected": names,
                "mean_abs_rho_off_diag": f"{mean_abs:.6f}",
                "cluster_match_accuracy": f"{acc:.4f}",
                "kmeans_inertia": f"{km['inertia']:.2f}",
            }
        )
    # with open(DIR / "rank_levels_summary.csv", "w", encoding="utf-8", newline="") as file:
    #     writer = csv.DictWriter(
    #         file,
    #         fieldnames=[
    #             "rank_levels",
    #             "selected_count",
    #             "selected",
    #             "mean_abs_rho_off_diag",
    #             "cluster_match_accuracy",
    #             "kmeans_inertia",
    #         ],
    #     )
    #     writer.writeheader()
    #     writer.writerows(rows)
    return rows


def print_ranking_mismatch_diagnosis(sweep_rows, target_n):
    best = None
    for row in sweep_rows:
        c = int(row["selected_count"])
        d = abs(c - target_n)
        if best is None or d < best[0]:
            best = (d, row)
    candidates = [r for r in sweep_rows if int(r["selected_count"]) == target_n]
    k_ok = ", ".join(r["rank_levels"] for r in candidates) if candidates else "—"
    print(f"отбор != {target_n}: ближайший K={best[1]['rank_levels']} n={best[1]['selected_count']}")
    print(f"K с n={target_n}: {k_ok}")

"""

def split_sample_indices(n, train_ratio, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    split_point = int(n * train_ratio)
    return indices[:split_point], indices[split_point:]


def empirical_distribution(sample):
    sample = np.sort(np.asarray(sample, dtype=float))
    if len(sample) == 0:
        return sample, np.array([], dtype=float)
    f = np.arange(1, len(sample) + 1) / len(sample)
    return sample, f


def max_distribution_difference(sample_1, sample_2):
    sample_1 = np.sort(np.asarray(sample_1, dtype=float))
    sample_2 = np.sort(np.asarray(sample_2, dtype=float))
    if len(sample_1) == 0 or len(sample_2) == 0:
        return float("nan"), float("nan")
    z = np.sort(np.unique(np.concatenate([sample_1, sample_2])))
    f1 = np.searchsorted(sample_1, z, side="right") / len(sample_1)
    f2 = np.searchsorted(sample_2, z, side="right") / len(sample_2)
    diff = np.abs(f1 - f2)
    i = int(np.argmax(diff))
    return float(diff[i]), float(z[i])


def estimate_alpha_beta(ksi0, ksi1):
    ksi0 = np.maximum(np.asarray(ksi0, dtype=float), LOG_FLOOR)
    ksi1 = np.maximum(np.asarray(ksi1, dtype=float), LOG_FLOOR)
    lnx = np.log(ksi0)
    lny = np.log(ksi1)
    mean_x = float(np.mean(lnx))
    mean_y = float(np.mean(lny))
    var_x = float(np.var(lnx, ddof=1)) if len(lnx) > 1 else 0.0
    var_y = float(np.var(lny, ddof=1)) if len(lny) > 1 else 0.0
    if var_x <= 0.0:
        beta = 1.0
    else:
        beta = math.sqrt(var_y / var_x)
    alpha = math.exp(mean_y - beta * mean_x)
    return alpha, beta


def phi(x, alpha, beta):
    x = np.maximum(np.asarray(x, dtype=float), LOG_FLOOR)
    return alpha * np.power(x, beta)


def evaluate_link_train_test(ksi0, ksi1, train_ratio=0.8, seed=RANDOM_SEED):
    ksi0 = np.asarray(ksi0, dtype=float)
    ksi1 = np.asarray(ksi1, dtype=float)
    n = min(len(ksi0), len(ksi1))
    ksi0 = ksi0[:n]
    ksi1 = ksi1[:n]
    if n < 4:
        return None
    tr, te = split_sample_indices(n, train_ratio, seed)
    k0_tr = ksi0[tr]
    k1_tr = ksi1[tr]
    k0_te = ksi0[te]
    k1_te = ksi1[te]
    alpha, beta = estimate_alpha_beta(k0_tr, k1_tr)
    d_raw_tr, z_raw_tr = max_distribution_difference(k0_tr, k1_tr)
    d_model_tr, z_model_tr = max_distribution_difference(phi(k0_tr, alpha, beta), k1_tr)
    d_raw_te, z_raw_te = max_distribution_difference(k0_te, k1_te)
    d_model_te, z_model_te = max_distribution_difference(phi(k0_te, alpha, beta), k1_te)
    d_raw_all, _ = max_distribution_difference(ksi0, ksi1)
    d_model_all, _ = max_distribution_difference(phi(ksi0, alpha, beta), ksi1)
    xm = np.asarray(phi(ksi0, alpha, beta), dtype=float)
    x1 = np.asarray(ksi1, dtype=float)
    p_ks_model_full = float(ks_2samp(xm, x1).pvalue)
    return {
        "alpha": alpha,
        "beta": beta,
        "d_raw_train": d_raw_tr,
        "z_raw_train": z_raw_tr,
        "d_model_train": d_model_tr,
        "z_model_train": z_model_tr,
        "d_raw_test": d_raw_te,
        "z_raw_test": z_raw_te,
        "d_model_test": d_model_te,
        "z_model_test": z_model_te,
        "d_raw_full": d_raw_all,
        "d_model_full": d_model_all,
        "p_ks_model_full": p_ks_model_full,
        "improve_train": (d_raw_tr / d_model_tr) if d_model_tr and d_model_tr > 0 else float("inf"),
        "improve_test": (d_raw_te / d_model_te) if d_model_te and d_model_te > 0 else float("inf"),
    }


def save_link_functions_csv(path, rows):
    fieldnames = [
        "from_feature",
        "to_feature",
        "spearman_rho",
        "alpha",
        "beta",
        "d_raw_train",
        "d_model_train",
        "improve_train",
        "d_raw_test",
        "d_model_test",
        "improve_test",
        "d_raw_full",
        "d_model_full",
        "p_ks_model_full",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_link_functions_txt(path, rows):
    lines = []
    for row in rows:
        fr = row.get("from_feature", "")
        to = row.get("to_feature", "")
        lines.append(f"=== {fr} -> {to} ===")
        lines.append(f"spearman_rho: {row.get('spearman_rho', '')}")
        lines.append(f"alpha: {row.get('alpha', '')}")
        lines.append(f"beta: {row.get('beta', '')}")
        lines.append(f"d_raw_train: {row.get('d_raw_train', '')}")
        lines.append(f"d_model_train: {row.get('d_model_train', '')}")
        lines.append(f"improve_train: {row.get('improve_train', '')}")
        lines.append(f"d_raw_test: {row.get('d_raw_test', '')}")
        lines.append(f"d_model_test: {row.get('d_model_test', '')}")
        lines.append(f"improve_test: {row.get('improve_test', '')}")
        lines.append(f"d_raw_full: {row.get('d_raw_full', '')}")
        lines.append(f"d_model_full: {row.get('d_model_full', '')}")
        lines.append(f"p_ks_model_full: {row.get('p_ks_model_full', '')}")
        lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines))


def plot_link_ecdf(from_name, to_name, ksi0, ksi1, alpha, beta, path):
    ksi0 = np.maximum(np.asarray(ksi0, dtype=float), LOG_FLOOR)
    ksi1 = np.maximum(np.asarray(ksi1, dtype=float), LOG_FLOOR)
    x0, f0 = empirical_distribution(ksi0)
    x1, f1 = empirical_distribution(ksi1)
    xf, ff = empirical_distribution(phi(ksi0, alpha, beta))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(x0, f0, where="post", label=f"F({from_name})")
    ax.step(x1, f1, where="post", label=f"F({to_name})")
    ax.step(xf, ff, where="post", label=f"F(phi({from_name}))")
    ax.set_title(f"{from_name} -> {to_name}, alpha={alpha:.4g}, beta={beta:.4f}")
    ax.set_xlabel("z")
    ax.set_ylabel("F(z)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def link_dependency_confirmed(ev):
    p = ev["p_ks_model_full"]
    if not math.isfinite(p):
        return False
    return p >= KS_SIGNIFICANCE_LEVEL


def format_pair_dependency_report(xi1_name, xi2_name, alpha, beta, confirmed, p_ks):
    status = "\u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0434\u0438\u043b\u0430\u0441\u044c" if confirmed else "\u043d\u0435 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0434\u0438\u043b\u0430\u0441\u044c"
    return [
        f"{xi1_name} - {xi2_name}",
        f"alpha={alpha:.8g}, beta={beta:.8f}",
        status,
        "",
    ]


def main():
    data_csv = DIR / "diabetes.csv"
    if not data_csv.is_file():
        raise FileNotFoundError(f"Ожидается файл данных: {data_csv}")

    feature_names, data, _ = load_data(data_csv)
    coeff_matrix = spearman_correlation_matrix(data, EXPORT_RANK_LEVELS)
    title_matrix = f"Матрица корреляции по рангам с K={EXPORT_RANK_LEVELS} уровнями"
    draw_spearman_matrix_table(feature_names, coeff_matrix, DIR / "spearman_matrix.png", title=title_matrix)
    strength = strength_matrix(coeff_matrix)
    selected_indices = select_features(strength, SPEARMAN_EDGE_THRESHOLD, feature_names)
    selected = ", ".join(feature_names[i] for i in selected_indices)
    print(f"отобранные признаки: {selected}")
    selected_set = set(selected_indices)
    non_selected = [idx for idx in range(len(feature_names)) if idx not in selected_set]
    not_kept = ", ".join(feature_names[i] for i in non_selected)
    print(f"не отобраны: {not_kept}")
    cross_pairs = []
    for s in selected_indices:
        for n in non_selected:
            cross_pairs.append((n, s))
    link_rows = []
    report_lines = []
    plots_dir = DIR / "link_ecdf"
    plots_dir.mkdir(exist_ok=True)
    confirmed_count = 0
    for a, b in cross_pairs:
        ksi0 = data[:, a]
        ksi1 = data[:, b]
        ev = evaluate_link_train_test(ksi0, ksi1)
        if ev is None:
            continue
        row = {
            "from_feature": feature_names[a],
            "to_feature": feature_names[b],
            "spearman_rho": f"{float(coeff_matrix[a, b]):.6f}",
            "alpha": f"{ev['alpha']:.8g}",
            "beta": f"{ev['beta']:.8f}",
            "d_raw_train": f"{ev['d_raw_train']:.6f}",
            "d_model_train": f"{ev['d_model_train']:.6f}",
            "improve_train": f"{ev['improve_train']:.6f}",
            "d_raw_test": f"{ev['d_raw_test']:.6f}",
            "d_model_test": f"{ev['d_model_test']:.6f}",
            "improve_test": f"{ev['improve_test']:.6f}",
            "d_raw_full": f"{ev['d_raw_full']:.6f}",
            "d_model_full": f"{ev['d_model_full']:.6f}",
            "p_ks_model_full": f"{ev['p_ks_model_full']:.8f}",
        }
        link_rows.append(row)
        alpha = ev["alpha"]
        beta = ev["beta"]
        confirmed = link_dependency_confirmed(ev)
        if confirmed:
            confirmed_count += 1
        xi1_name = feature_names[b]
        xi2_name = feature_names[a]
        block = format_pair_dependency_report(
            xi1_name, xi2_name, alpha, beta, confirmed, ev["p_ks_model_full"]
        )
        report_lines.extend(block)
        for line in block:
            if line:
                print(line)
        print()
        fname = f"{feature_names[a]}__to__{feature_names[b]}.png".replace("/", "_")
        plot_link_ecdf(feature_names[a], feature_names[b], ksi0, ksi1, alpha, beta, plots_dir / fname)
    save_link_functions_csv(DIR / "link_functions.csv", link_rows)
    save_link_functions_txt(DIR / "link_functions.txt", link_rows)
    (DIR / "pair_link_report.txt").write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    #print(f"KS (p>={KS_SIGNIFICANCE_LEVEL}): подтвердилось {confirmed_count} из {len(link_rows)} направлений")


if __name__ == "__main__":
    main()
