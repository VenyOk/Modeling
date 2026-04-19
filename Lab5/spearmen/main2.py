import csv
import math
import random
from pathlib import Path
from statistics import NormalDist

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

DIR = Path(__file__).resolve().parent
DIABETES_CSV = DIR.parent / "diabetes.csv"

SPEARMAN_MATRIX_FILE = DIR / "spearman_matrix.csv"
SPEARMAN_MATRIX_PNG = DIR / "spearman_matrix.png"
STRONG_RELATION_THRESHOLD = 0.45
USE_NORMAL_SIGNIFICANCE_TEST = True
SIGNIFICANCE_ALPHA = 0.05
CLUSTERS_CSV = DIR / "clusters.csv"
CLUSTER_SUMMARY_CSV = DIR / "cluster_summary.csv"
TRUTH_TABLE_FILE = DIR / "truth_table.csv"
KMEANS_PLOT = DIR / "kmeans_clusters_2d.png"
RANK_LEVELS_SUMMARY_CSV = DIR / "rank_levels_summary.csv"
RANK_LEVELS_FOR_SWEEP = (2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, None)
EXPORT_RANK_LEVELS = 16


def load_data(path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    feature_names = [name for name in reader.fieldnames if name != "Outcome"]
    data = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    outcome = np.array([int(row["Outcome"]) for row in rows], dtype=int)
    return feature_names, data, outcome


def median_binarize(data):
    medians = np.median(data, axis=0)
    return medians, data > medians


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


def spearman_pairwise_normal_pvalues(coeff_matrix, n_samples):
    p = coeff_matrix.shape[0]
    pvals = np.ones((p, p), dtype=float)
    if n_samples < 3:
        return pvals
    scale = math.sqrt(n_samples - 1)
    nd = NormalDist(0, 1)
    for i in range(p):
        for j in range(i + 1, p):
            rho = float(coeff_matrix[i, j])
            z = scale * rho
            pv = 2 * (1 - nd.cdf(abs(z)))
            pvals[i, j] = pvals[j, i] = pv
    return pvals


def strength_matrix_significant_bonferroni(coeff_matrix, n_samples, alpha=SIGNIFICANCE_ALPHA):
    p = coeff_matrix.shape[0]
    strength = np.zeros((p, p), dtype=float)
    m = p * (p - 1) // 2
    if m == 0 or n_samples < 3:
        return strength, 0, float("nan"), float("nan")
    alpha_pair = alpha / m
    z_crit = NormalDist(0, 1).inv_cdf(1 - alpha_pair / 2)
    pvals = spearman_pairwise_normal_pvalues(coeff_matrix, n_samples)
    count = 0
    for i in range(p):
        for j in range(i + 1, p):
            if pvals[i, j] < alpha_pair:
                v = abs(float(coeff_matrix[i, j]))
                strength[i, j] = strength[j, i] = v
                count += 1
    return strength, count, alpha_pair, z_crit


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
            label=f"Class {cluster_id}",
        )
    axis.scatter(
        centroid_projection[:, 0],
        centroid_projection[:, 1],
        s=160,
        c="#000000",
        marker="X",
        label="Centroids",
    )
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title("k-means clusters in PCA 2D projection (Spearman selection)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_clusters_csv(data, feature_names, selected_indices, labels_raw, labels_aligned, outcome, path):
    selected_names = [feature_names[idx] for idx in selected_indices]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["object_id"] + selected_names + ["cluster_raw", "cluster_aligned", "Outcome"])
        for i in range(len(data)):
            values = [f"{data[i, idx]:.6f}" for idx in selected_indices]
            writer.writerow([i + 1] + values + [int(labels_raw[i]), int(labels_aligned[i]), int(outcome[i])])


def save_cluster_summary_csv(feature_names, selected_indices, medians, data, labels_aligned, path):
    selected_names = [feature_names[idx] for idx in selected_indices]
    header = ["class", "count"]
    for name in selected_names:
        header.append(f"{name}_centroid")
        header.append(f"{name}_vs_median")
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
                direction = "above_median" if value > medians[feature_idx] else "below_median"
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


def print_text_output(
    feature_names,
    selected_indices,
    truth_table_counts,
    cluster_counts,
    selection_note,
):
    selected = ", ".join(feature_names[idx] for idx in selected_indices)
    print(f"Отбор признаков: {selection_note}")
    print(f"Выбранные признаки: {selected}")
    print(f"Метки Y в таблице истинности -> 0: {truth_table_counts[0]}, 1: {truth_table_counts[1]}")
    print(f"Метки кластеров -> 0: {cluster_counts[0]}, 1: {cluster_counts[1]}")


def strength_and_edge_threshold(coeff_matrix, n_samples):
    if USE_NORMAL_SIGNIFICANCE_TEST:
        strength, sig_pairs, alpha_pair, z_crit = strength_matrix_significant_bonferroni(
            coeff_matrix, n_samples, SIGNIFICANCE_ALPHA
        )
        return strength, 1e-12, sig_pairs
    strength = strength_matrix(coeff_matrix)
    return strength, STRONG_RELATION_THRESHOLD, None


def sweep_rank_levels(feature_names, data, outcome):
    n_samples = data.shape[0]
    p = len(feature_names)
    rows = []
    print("--- Svodka: chislo urovnej ranzhirovki K (ravnochastotno 1..K po pozicijam v sortirovke) ---")
    for num_levels in RANK_LEVELS_FOR_SWEEP:
        coeff = spearman_correlation_matrix(data, num_levels)
        strength, edge_thr, sig_pairs = strength_and_edge_threshold(coeff, n_samples)
        selected = select_features(strength, edge_thr, feature_names)
        names = ";".join(feature_names[i] for i in selected)
        off = coeff.astype(float).copy()
        np.fill_diagonal(off, 0.0)
        mean_abs = float(np.abs(off).sum() / max(p * (p - 1), 1))
        sel_points = data[:, selected]
        std_pts = standardize(sel_points)
        km = kmeans(std_pts, k=2, n_init=30, max_iterations=300, seed=42)
        _, acc = align_clusters_to_outcome(km["labels"], outcome)
        label = "full" if num_levels is None else str(int(num_levels))
        rows.append(
            {
                "rank_levels": label,
                "selected_count": len(selected),
                "selected": names,
                "significant_pairs": "" if sig_pairs is None else int(sig_pairs),
                "mean_abs_rho_off_diag": f"{mean_abs:.6f}",
                "cluster_match_accuracy": f"{acc:.4f}",
                "kmeans_inertia": f"{km['inertia']:.2f}",
            }
        )
        sp = "-" if sig_pairs is None else str(sig_pairs)
        print(
            f"K={label:>5}  n_sel={len(selected):2}  signif_pairs={sp:>3}  "
            f"mean|rho|_offd={mean_abs:.4f}  cluster_vs_outcome={acc:.4f}  -> {names}"
        )
    with open(RANK_LEVELS_SUMMARY_CSV, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank_levels",
                "selected_count",
                "selected",
                "significant_pairs",
                "mean_abs_rho_off_diag",
                "cluster_match_accuracy",
                "kmeans_inertia",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Svodka zapisana: {RANK_LEVELS_SUMMARY_CSV}")


def main():
    if not DIABETES_CSV.is_file():
        raise FileNotFoundError(f"Ожидается файл данных: {DIABETES_CSV}")

    feature_names, data, outcome = load_data(DIABETES_CSV)
    medians, binary_data = median_binarize(data)
    n_samples = data.shape[0]
    sweep_rank_levels(feature_names, data, outcome)
    coeff_matrix = spearman_correlation_matrix(data, EXPORT_RANK_LEVELS)
    save_matrix_csv(feature_names, coeff_matrix, SPEARMAN_MATRIX_FILE)
    title_matrix = (
        "Матрица корреляции Спирмена (полные ранги 1..n)"
        if EXPORT_RANK_LEVELS is None
        else f"Матрица корреляции по рангам с K={EXPORT_RANK_LEVELS} уровнями"
    )
    draw_spearman_matrix_table(feature_names, coeff_matrix, SPEARMAN_MATRIX_PNG, title=title_matrix)
    if USE_NORMAL_SIGNIFICANCE_TEST:
        strength, sig_pairs, alpha_pair, z_crit = strength_matrix_significant_bonferroni(
            coeff_matrix, n_samples, SIGNIFICANCE_ALPHA
        )
        m = len(feature_names) * (len(feature_names) - 1) // 2
        print(
            f"Проверка из лекции: z = sqrt(n-1)*rho ~ N(0,1) при H0, n={n_samples}, пар m={m}, "
            f"alpha={SIGNIFICANCE_ALPHA}, Бонферрони alpha/m={alpha_pair:.6g}, |z|_крит={z_crit:.4f}, "
            f"значимых пар: {sig_pairs}"
        )
        edge_threshold = 1e-12
        selection_note = (
            f"корреляция Спирмена; рёбра графа только при p < alpha/m "
            f"(двусторонний нормальный тест по z=sqrt(n-1)*rho)"
        )
    else:
        strength = strength_matrix(coeff_matrix)
        edge_threshold = STRONG_RELATION_THRESHOLD
        selection_note = f"корреляция Спирмена; |rho| >= {STRONG_RELATION_THRESHOLD} для ребра графа"
    selected_indices = select_features(strength, edge_threshold, feature_names)
    selected_points = data[:, selected_indices]
    standardized_points = standardize(selected_points)
    km = kmeans(standardized_points, k=2, n_init=30, max_iterations=300, seed=42)
    labels_raw = km["labels"]
    labels_aligned, _ = align_clusters_to_outcome(labels_raw, outcome)

    best = {
        "selected_indices": selected_indices,
        "labels_raw": labels_raw,
        "labels_aligned": labels_aligned,
        "centroids_standardized": km["centroids"],
    }

    save_clusters_csv(
        data,
        feature_names,
        best["selected_indices"],
        best["labels_raw"],
        best["labels_aligned"],
        outcome,
        CLUSTERS_CSV,
    )
    save_cluster_summary_csv(
        feature_names,
        best["selected_indices"],
        medians,
        data,
        best["labels_aligned"],
        CLUSTER_SUMMARY_CSV,
    )
    truth_table_rows = build_truth_table_rows(binary_data, best["labels_aligned"])
    save_truth_table_csv(
        feature_names,
        truth_table_rows,
        TRUTH_TABLE_FILE,
    )
    draw_cluster_projection(
        standardized_points,
        best["labels_aligned"],
        best["centroids_standardized"],
        KMEANS_PLOT,
    )
    truth_table_counts = count_truth_table_labels(truth_table_rows)
    cluster_counts = count_binary_labels(best["labels_aligned"])
    print_text_output(
        feature_names,
        best["selected_indices"],
        truth_table_counts,
        cluster_counts,
        selection_note,
    )


if __name__ == "__main__":
    main()
