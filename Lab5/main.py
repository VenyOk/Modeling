import csv
import random

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

ASSOCIATION_MATRIX_FILE = "association_matrix.csv"
STRONG_RELATION_THRESHOLD = 0.45
TRUTH_TABLE_FILE = "truth_table.csv"
<<<<<<< HEAD
LOGIC_FORMS_FILE = "extra_features_sdnf.txt"
=======
>>>>>>> 47d70492a5c5d567c10bfd3190cd873831dc081a
FIXED_SELECTED_FEATURES = ["Glucose", "SkinThickness", "DiabetesPedigreeFunction", "Age"]


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


def load_matrix_csv(path, expected_feature_names):
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        file_features = header[1:]
        if file_features != expected_feature_names:
            raise ValueError("association_matrix.csv does not match feature order in diabetes.csv")
        rows = list(reader)
    row_features = [row[0] for row in rows]
    if row_features != expected_feature_names:
        raise ValueError("association_matrix.csv row labels do not match diabetes.csv")
    matrix = np.array([[float(x) for x in row[1:]] for row in rows], dtype=float)
    return matrix


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
    axis.set_title("k-means clusters in PCA 2D projection")
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


def get_fixed_selected_indices(feature_names):
    index_by_name = {name: idx for idx, name in enumerate(feature_names)}
    return [index_by_name[name] for name in FIXED_SELECTED_FEATURES]


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
<<<<<<< HEAD
    variable_names = [f"x{i}" for i in range(1, len(feature_names) + 1)]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(variable_names + ["y"])
=======
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(feature_names + ["Y"])
>>>>>>> 47d70492a5c5d567c10bfd3190cd873831dc081a
        writer.writerows(truth_table_rows)


def count_binary_labels(labels):
    labels = np.asarray(labels, dtype=int)
    return int((labels == 0).sum()), int((labels == 1).sum())


def count_truth_table_labels(truth_table_rows):
    labels = [row[-1] for row in truth_table_rows]
    return count_binary_labels(labels)


def merge_terms(left, right):
    merged = []
    differences = 0
    for left_bit, right_bit in zip(left, right):
        if left_bit == right_bit:
            merged.append(left_bit)
            continue
        if left_bit == -1 or right_bit == -1:
            return None
        differences += 1
        merged.append(-1)
        if differences > 1:
            return None
    if differences != 1:
        return None
    return tuple(merged)


def term_covers(term, pattern):
    return all(term_bit == -1 or term_bit == pattern_bit for term_bit, pattern_bit in zip(term, pattern))


def term_covers_any(term, patterns):
    return any(term_covers(term, pattern) for pattern in patterns)


def term_literal_count(term):
    return sum(bit != -1 for bit in term)


def format_dnf(feature_names, terms):
    if not terms:
        return "0"
    variable_names = [f"x{i}" for i in range(1, len(feature_names) + 1)]
    if any(all(bit == -1 for bit in term) for term in terms):
        return "1"
    formatted_terms = []
    for pattern in sorted(terms, key=lambda term: (term_literal_count(term), term)):
        literals = [
            name if bit == 1 else f"!{name}"
            for name, bit in zip(variable_names, pattern)
            if bit != -1
        ]
        if literals:
            formatted_terms.append("(" + " & ".join(literals) + ")")
        else:
            formatted_terms.append("1")
    return " | ".join(formatted_terms)


<<<<<<< HEAD
def build_sdnf_terms(truth_table_rows):
    return {
        tuple(int(bit) for bit in row[:-1])
        for row in truth_table_rows
        if int(row[-1]) == 1
    }


=======
>>>>>>> 47d70492a5c5d567c10bfd3190cd873831dc081a
def build_dnf_terms(truth_table_rows):
    positive_patterns = {
        tuple(row[:-1])
        for row in truth_table_rows
        if int(row[-1]) == 1
    }
    negative_patterns = {
        tuple(row[:-1])
        for row in truth_table_rows
        if int(row[-1]) == 0
    }
    if not positive_patterns:
<<<<<<< HEAD
        return set()
=======
        return "0"
>>>>>>> 47d70492a5c5d567c10bfd3190cd873831dc081a
    current_terms = set(positive_patterns)
    prime_implicants = set()
    while current_terms:
        current_list = sorted(current_terms)
        used_terms = set()
        next_terms = set()
        for left_index in range(len(current_list)):
            for right_index in range(left_index + 1, len(current_list)):
                merged = merge_terms(current_list[left_index], current_list[right_index])
                if merged is None or term_covers_any(merged, negative_patterns):
                    continue
                used_terms.add(current_list[left_index])
                used_terms.add(current_list[right_index])
                next_terms.add(merged)
        prime_implicants.update(term for term in current_terms if term not in used_terms)
        current_terms = next_terms
    prime_implicants = {
        term for term in prime_implicants
        if term_covers_any(term, positive_patterns)
    }
    coverage = {
        pattern: {term for term in prime_implicants if term_covers(term, pattern)}
        for pattern in positive_patterns
    }
    selected_terms = set()
    covered_patterns = set()
    for pattern, covering_terms in coverage.items():
        if len(covering_terms) == 1:
            selected_terms.update(covering_terms)
    for term in selected_terms:
        covered_patterns.update(pattern for pattern in positive_patterns if term_covers(term, pattern))
    remaining_patterns = positive_patterns - covered_patterns
    available_terms = set(prime_implicants)
    while remaining_patterns:
        best_term = max(
            available_terms,
            key=lambda term: (
                sum(term_covers(term, pattern) for pattern in remaining_patterns),
                -term_literal_count(term),
                tuple(-1 if bit == -1 else bit for bit in term),
            ),
        )
        selected_terms.add(best_term)
        covered_now = {pattern for pattern in remaining_patterns if term_covers(best_term, pattern)}
        remaining_patterns -= covered_now
        available_terms.discard(best_term)
    reduced_terms = set(selected_terms)
    for term in sorted(selected_terms, key=lambda item: (term_literal_count(item), item), reverse=True):
        other_terms = reduced_terms - {term}
        if positive_patterns and all(any(term_covers(other, pattern) for other in other_terms) for pattern in positive_patterns if term_covers(term, pattern)):
            reduced_terms.discard(term)
    return reduced_terms


<<<<<<< HEAD
def save_logic_forms(path, sdnf, dnf):
    with open(path, "w", encoding="utf-8", newline="") as file:
        file.write(f"SDNF: {sdnf}\n")
        file.write(f"DNF: {dnf}\n")


=======
>>>>>>> 47d70492a5c5d567c10bfd3190cd873831dc081a
def evaluate_dnf_terms(terms, binary_data):
    labels = []
    for bits in binary_data.astype(int):
        pattern = tuple(int(bit) for bit in bits)
        label = 1 if any(term_covers(term, pattern) for term in terms) else 0
        labels.append(label)
    return np.array(labels, dtype=int)


def compare_labels(predicted_labels, reference_labels):
    predicted_labels = np.asarray(predicted_labels, dtype=int)
    reference_labels = np.asarray(reference_labels, dtype=int)
    matches = int((predicted_labels == reference_labels).sum())
    total = int(len(reference_labels))
    accuracy = matches / total if total else 0.0
    return matches, total, accuracy


def print_text_output(feature_names, selected_indices, truth_table_counts, cluster_counts, dnf_counts, dnf_match_stats, dnf):
    selected = ", ".join(feature_names[idx] for idx in selected_indices)
<<<<<<< HEAD
    print(f"Выбранные признаки: {selected}")
    print(f"Метки Y в таблице истинности -> 0: {truth_table_counts[0]}, 1: {truth_table_counts[1]}")
    print(f"Метки кластеров -> 0: {cluster_counts[0]}, 1: {cluster_counts[1]}")
    print(f"Метки ДНФ -> 0: {dnf_counts[0]}, 1: {dnf_counts[1]}")
    print(f"Сравнение ДНФ и кластеров -> совпадения: {dnf_match_stats[0]}/{dnf_match_stats[1]}, точность: {dnf_match_stats[2]:.4f}")
    print(f"ДНФ: {dnf}")
=======
    print(f"Selected features: {selected}")
    print(f"Truth table Y -> 0: {truth_table_counts[0]}, 1: {truth_table_counts[1]}")
    print(f"Cluster labels -> 0: {cluster_counts[0]}, 1: {cluster_counts[1]}")
    print(f"DNF labels -> 0: {dnf_counts[0]}, 1: {dnf_counts[1]}")
    print(f"DNF vs cluster labels -> matches: {dnf_match_stats[0]}/{dnf_match_stats[1]}, accuracy: {dnf_match_stats[2]:.4f}")
    print(f"DNF: {dnf}")
>>>>>>> 47d70492a5c5d567c10bfd3190cd873831dc081a


def main():
    feature_names, data, outcome = load_data("diabetes.csv")
    medians, binary_data = median_binarize(data)
    selected_indices = get_fixed_selected_indices(feature_names)
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
        "clusters.csv",
    )
    save_cluster_summary_csv(
        feature_names,
        best["selected_indices"],
        medians,
        data,
        best["labels_aligned"],
        "cluster_summary.csv",
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
        "kmeans_clusters_2d.png",
    )
    truth_table_counts = count_truth_table_labels(truth_table_rows)
    cluster_counts = count_binary_labels(best["labels_aligned"])
<<<<<<< HEAD
    sdnf_terms = build_sdnf_terms(truth_table_rows)
    sdnf = format_dnf(feature_names, sdnf_terms)
    dnf_terms = build_dnf_terms(truth_table_rows)
    dnf = format_dnf(feature_names, dnf_terms)
    save_logic_forms(LOGIC_FORMS_FILE, sdnf, dnf)
=======
    dnf_terms = build_dnf_terms(truth_table_rows)
    dnf = format_dnf(feature_names, dnf_terms)
>>>>>>> 47d70492a5c5d567c10bfd3190cd873831dc081a
    dnf_labels = evaluate_dnf_terms(dnf_terms, binary_data)
    dnf_counts = count_binary_labels(dnf_labels)
    dnf_match_stats = compare_labels(dnf_labels, best["labels_aligned"])
    print_text_output(
        feature_names,
        best["selected_indices"],
        truth_table_counts,
        cluster_counts,
        dnf_counts,
        dnf_match_stats,
        dnf,
    )


if __name__ == "__main__":
    main()

