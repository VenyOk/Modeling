import csv
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def load_data(path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    feature_names = [name for name in reader.fieldnames if name != "Outcome"]
    data = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    return feature_names, data


def median_binarize(data):
    medians = np.median(data, axis=0)
    return data > medians


def contingency_counts(left, right):
    a = int(np.logical_and(left, right).sum())
    b = int(np.logical_and(left, ~right).sum())
    c = int(np.logical_and(~left, right).sum())
    d = int(np.logical_and(~left, ~right).sum())
    return a, b, c, d


def association_matrix(binary):
    count = binary.shape[1]
    matrix = np.eye(count, dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            a, b, c, d = contingency_counts(binary[:, i], binary[:, j])
            denominator = a * d + b * c
            value = (a * d - b * c) / denominator if denominator else 0.0
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def colligation_matrix(binary):
    objects_count = binary.shape[0]
    feature_count = binary.shape[1]
    matrix = np.ones((feature_count, feature_count), dtype=float)
    probabilities = binary.sum(axis=0) / objects_count
    for i in range(feature_count):
        probability_a = float(probabilities[i])
        for j in range(feature_count):
            probability_b = float(probabilities[j])
            joint_probability = float(np.logical_and(binary[:, i], binary[:, j]).sum() / objects_count)
            probability_a_given_b = joint_probability / probability_b if probability_b else 0.0
            matrix[i, j] = probability_a_given_b / probability_a if probability_a else 0.0
    return matrix


def pearson_contingency_matrix(binary):
    count = binary.shape[1]
    matrix = np.eye(count, dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            a, b, c, d = contingency_counts(binary[:, i], binary[:, j])
            denominator = math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
            value = (a * d - b * c) / denominator if denominator else 0.0
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def yule_matrix(binary):
    count = binary.shape[1]
    matrix = np.eye(count, dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            a, b, c, d = contingency_counts(binary[:, i], binary[:, j])
            ad = math.sqrt(a * d)
            bc = math.sqrt(b * c)
            denominator = ad + bc
            value = (ad - bc) / denominator if denominator else 0.0
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def save_matrix_csv(feature_names, matrix, path):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["feature"] + feature_names)
        for feature_name, row in zip(feature_names, matrix):
            writer.writerow([feature_name] + [f"{value:.6f}" for value in row])


def draw_matrix_table(feature_names, matrix, coefficient_name, title, path):
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
    center = 1.0 if coefficient_name == "colligation" else 0.0
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


def main():
    feature_names, data = load_data("diabetes.csv")
    binary = median_binarize(data)
    matrices = {
        "association": association_matrix(binary),
        "colligation": colligation_matrix(binary),
        "pearson_contingency": pearson_contingency_matrix(binary),
        "yule": yule_matrix(binary),
    }
    titles = {
        "association": "Матрица коэффициентов ассоциации",
        "colligation": "Матрица коэффициентов взаимосвязи",
        "pearson_contingency": "Матрица коэффициентов контингенции Пирсона",
        "yule": "Матрица коэффициентов Юла",
    }
    for name, matrix in matrices.items():
        csv_path = f"{name}_matrix.csv"
        png_path = f"{name}_matrix.png"
        save_matrix_csv(feature_names, matrix, csv_path)
        draw_matrix_table(feature_names, matrix, name, titles[name], png_path)
    save_matrix_csv(feature_names, matrices["association"], "selected_coefficient_matrix.csv")
    draw_matrix_table(
        feature_names,
        matrices["association"],
        "association",
        "Selected coefficient matrix: association",
        "selected_coefficient_matrix.png",
    )


if __name__ == "__main__":
    main()
