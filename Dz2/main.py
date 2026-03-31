import random
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    raise SystemExit("Install matplotlib: pip install matplotlib") from error


TRIALS = 1000
P = [0, 0.25, 0.5, 0.75, 1]
GRAPH_TRIALS = [10, 100, 1000, 10000, 100000, 1000000]
PLOT_PATH = Path(__file__).with_name("average_error_plot.png")


def theoretical_probability(n, k, p_save):
    p_stay = 1 / n
    p_switch = (n - 1) / (n * (n - k - 1))
    return p_save * p_stay + (1 - p_save) * p_switch


def experiment(n, k, p_save, trials):
    wins = 0

    for _ in range(trials):
        prize = random.randrange(n)
        first_choice = random.randrange(n)

        doors_for_host = []
        for door in range(n):
            if door != first_choice and door != prize:
                doors_for_host.append(door)

        opened_doors = set(random.sample(doors_for_host, k))

        if random.random() < p_save:
            final_choice = first_choice
        else:
            available_doors = []
            for door in range(n):
                if door != first_choice and door not in opened_doors:
                    available_doors.append(door)
            final_choice = random.choice(available_doors)

        if final_choice == prize:
            wins += 1

    return wins / trials


def fast_experiment(probability, trials, rng):
    wins = rng.binomialvariate(trials, probability)
    return wins / trials


def average_error_for_trials(trials):
    rng = random.Random(67)
    total_error = 0
    count = 0

    for n in range(3, 11):
        for k in range(1, n - 1):
            for p_save in P:
                theory = theoretical_probability(n, k, p_save)
                exp = fast_experiment(theory, trials, rng)
                total_error += abs(theory - exp)
                count += 1

    return total_error / count


def write_average_error_plot(path):
    points = []

    for trials in GRAPH_TRIALS:
        average_error = average_error_for_trials(trials)
        points.append((trials, average_error))

    x_values = [trials for trials, _ in points]
    y_values = [average_error for _, average_error in points]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o")
    plt.xscale("log")
    plt.xlabel("\u0427\u0438\u0441\u043b\u043e \u043f\u043e\u043f\u044b\u0442\u043e\u043a")
    plt.ylabel("\u041e\u0448\u0438\u0431\u043a\u0430")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return points


def monty_hall_check(trials):
    print("Monty Hall: N = 3, K = 1")
    print(f"{'strategy':<12}{'theory':>12}{'experiment':>14}{'error':>12}")

    for name, p_save in [("stay", 1), ("switch", 0)]:
        theory = theoretical_probability(3, 1, p_save)
        exp = experiment(3, 1, p_save, trials)
        error = abs(theory - exp)
        print(f"{name:<12}{theory:>12.6f}{exp:>14.6f}{error:>12.6f}")

    print()


def main():
    random.seed(67)

    monty_hall_check(TRIALS)

    print(f"{'N':>3}{'K':>4}{'p_save':>10}{'theory':>12}{'experiment':>14}{'error':>12}")

    for n in range(3, 11):
        for k in range(1, n - 1):
            for p_save in P:
                theory = theoretical_probability(n, k, p_save)
                exp = experiment(n, k, p_save, TRIALS)
                error = abs(theory - exp)
                print(f"{n:>3}{k:>4}{p_save:>10.2f}{theory:>12.6f}{exp:>14.6f}{error:>12.6f}")

    print()
    print("Average error by TRIALS")
    average_errors = write_average_error_plot(PLOT_PATH)
    print(f"{'TRIALS':>10}{'avg_error':>14}")
    for trials, average_error in average_errors:
        print(f"{trials:>10}{average_error:>14.6f}")


if __name__ == "__main__":
    main()
