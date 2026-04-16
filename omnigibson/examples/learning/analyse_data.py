import argparse
import ast
import re
from pathlib import Path
from statistics import mean


DEFAULT_LOG_PATH = "/home/user/Desktop/wq/try/first/more_metric.log"


def safe_mean(values):
    return mean(values) if values else 0.0


def parse_eval_log(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "{" not in line:
            continue

        raw_dict = line[line.index("{") :]
        raw_dict = re.sub(r"tensor\(([^)]+)\)", r"\1", raw_dict)

        try:
            data = ast.literal_eval(raw_dict)
        except Exception:
            continue

        rows.append(data)

    return rows


def summarize_overall(rows):
    total = len(rows)
    success_count = sum(1 for row in rows if row["success"])
    timeout_count = sum(1 for row in rows if row["step"] >= 500)
    no_grasp_count = sum(1 for row in rows if not row["grasp_events"])
    late_first_grasp_count = sum(
        1
        for row in rows
        if row["first_grasp_step"] is not None and row["first_grasp_step"] > 100
    )
    rows_with_release = [row for row in rows if row["release_events"]]
    released_before_target_count = sum(
        1 for row in rows_with_release if row["released_before_target"]
    )

    arrival_ratios = [
        row["arrival_num"] / row["obj_num"] for row in rows if row["obj_num"] > 0
    ]
    potential_deltas = [
        float(row["fini_potential"]) - float(row["init_potential"]) for row in rows
    ]

    return {
        "episode_count": total,
        "success_rate": success_count / total if total else 0.0,
        "success_count": success_count,
        "timeout_count": timeout_count,
        "avg_step": safe_mean([row["step"] for row in rows]),
        "avg_obj_num": safe_mean([row["obj_num"] for row in rows]),
        "avg_arrival_ratio": safe_mean(arrival_ratios),
        "avg_potential_delta": safe_mean(potential_deltas),
        "no_grasp_count": no_grasp_count,
        "no_grasp_ratio": no_grasp_count / total if total else 0.0,
        "late_first_grasp_count": late_first_grasp_count,
        "late_first_grasp_ratio": late_first_grasp_count / total if total else 0.0,
        "episodes_with_release_count": len(rows_with_release),
        "released_before_target_count": released_before_target_count,
        "released_before_target_ratio": (
            released_before_target_count / len(rows_with_release)
            if rows_with_release
            else 0.0
        ),
    }


def summarize_by_success(rows):
    result = {}
    for success_flag in [True, False]:
        subset = [row for row in rows if row["success"] == success_flag]
        subset_with_release = [row for row in subset if row["release_events"]]
        subset_with_grasp = [row for row in subset if row["grasp_events"]]
        first_grasp_steps = [
            row["first_grasp_step"]
            for row in subset
            if row["first_grasp_step"] is not None
        ]
        potential_deltas = [
            float(row["fini_potential"]) - float(row["init_potential"]) for row in subset
        ]
        total = len(subset)
        result[success_flag] = {
            "count": total,
            "avg_step": safe_mean([row["step"] for row in subset]),
            "avg_first_grasp_step": safe_mean(first_grasp_steps),
            "episodes_with_grasp_count": len(subset_with_grasp),
            "avg_grasp_count_among_grasp_episodes": safe_mean(
                [len(row["grasp_events"]) for row in subset_with_grasp]
            ),
            "no_grasp_ratio": (
                sum(1 for row in subset if not row["grasp_events"]) / total if total else 0.0
            ),
            "episodes_with_release_count": len(subset_with_release),
            "released_before_target_ratio": (
                sum(1 for row in subset_with_release if row["released_before_target"])
                / len(subset_with_release)
                if subset_with_release
                else 0.0
            ),
            "avg_potential_delta": safe_mean(potential_deltas),
        }
    return result


def summarize_by_obj_num(rows):
    result = {}
    for obj_num in sorted({row["obj_num"] for row in rows}):
        subset = [row for row in rows if row["obj_num"] == obj_num]
        total = len(subset)
        potential_deltas = [
            float(row["fini_potential"]) - float(row["init_potential"]) for row in subset
        ]
        result[obj_num] = {
            "count": total,
            "success_rate": (
                sum(1 for row in subset if row["success"]) / total if total else 0.0
            ),
            "avg_arrival_ratio": safe_mean(
                [row["arrival_num"] / row["obj_num"] for row in subset if row["obj_num"] > 0]
            ),
            "no_grasp_ratio": (
                sum(1 for row in subset if not row["grasp_events"]) / total if total else 0.0
            ),
            "avg_first_grasp_step": safe_mean(
                [
                    row["first_grasp_step"]
                    for row in subset
                    if row["first_grasp_step"] is not None
                ]
            ),
            "avg_potential_delta": safe_mean(potential_deltas),
        }
    return result


def summarize_release_events(rows):
    release_events = []
    for row in rows:
        release_events.extend(row["release_events"])

    total = len(release_events)
    if total == 0:
        return {
            "total_releases": 0,
            "release_on_target_ratio": 0.0,
            "release_farther_ratio": 0.0,
            "release_closer_ratio": 0.0,
            "release_unchanged_ratio": 0.0,
            "short_hold_ratio_le_5": 0.0,
            "avg_hold_steps": 0.0,
        }

    hold_steps = [
        event["steps_since_grasp"]
        for event in release_events
        if event["steps_since_grasp"] is not None
    ]

    return {
        "total_releases": total,
        "release_on_target_ratio": (
            sum(1 for event in release_events if event["released_on_target"]) / total
        ),
        "release_farther_ratio": (
            sum(1 for event in release_events if event["distance_change_vs_grasp"] == "farther")
            / total
        ),
        "release_closer_ratio": (
            sum(1 for event in release_events if event["distance_change_vs_grasp"] == "closer")
            / total
        ),
        "release_unchanged_ratio": (
            sum(1 for event in release_events if event["distance_change_vs_grasp"] == "unchanged")
            / total
        ),
        "short_hold_ratio_le_5": (
            sum(
                1
                for event in release_events
                if event["steps_since_grasp"] is not None and event["steps_since_grasp"] <= 5
            )
            / total
        ),
        "avg_hold_steps": safe_mean(hold_steps),
    }


def summarize_scene_categories(rows):
    total = len(rows)
    categories = {
        "no_navigation_count": 0,
        "near_but_no_grasp_count": 0,
        "grasped_but_not_success_count": 0,
        "grasped_and_success_count": 0,
    }

    for row in rows:
        success = row["success"]
        has_grasp_history = bool(row["grasp_events"])
        min_distances = list(row.get("min_distance_per_object", {}).values())
        min_distance = min(min_distances) if min_distances else None

        if success is True:
            categories["grasped_and_success_count"] += 1
        elif success is False and has_grasp_history:
            categories["grasped_but_not_success_count"] += 1
        elif success is False and min_distance is not None and min_distance < 1.5:
            categories["near_but_no_grasp_count"] += 1
        elif success is False and min_distance is not None and min_distance > 1.5:
            categories["no_navigation_count"] += 1

    categories["no_navigation_ratio"] = (
        categories["no_navigation_count"] / total if total else 0.0
    )
    categories["near_but_no_grasp_ratio"] = (
        categories["near_but_no_grasp_count"] / total if total else 0.0
    )
    categories["grasped_but_not_success_ratio"] = (
        categories["grasped_but_not_success_count"] / total if total else 0.0
    )
    categories["grasped_and_success_ratio"] = (
        categories["grasped_and_success_count"] / total if total else 0.0
    )
    categories["categorized_count"] = (
        categories["no_navigation_count"]
        + categories["near_but_no_grasp_count"]
        + categories["grasped_but_not_success_count"]
        + categories["grasped_and_success_count"]
    )
    categories["uncategorized_count"] = total - categories["categorized_count"]

    return categories


def print_section(title):
    print(f"\n{title}")
    print("-" * len(title))


def print_overall(summary):
    print_section("Overall")
    print(f"episode_count: {summary['episode_count']}")
    print(f"success_count: {summary['success_count']}")
    print(f"success_rate: {summary['success_rate']:.4f}")
    print(f"timeout_count: {summary['timeout_count']}")
    print(f"avg_step: {summary['avg_step']:.2f}")
    print(f"avg_obj_num: {summary['avg_obj_num']:.2f}")
    print(f"avg_arrival_ratio: {summary['avg_arrival_ratio']:.4f}")
    print(f"avg_potential_delta: {summary['avg_potential_delta']:.4f}")
    print(
        f"no_grasp: {summary['no_grasp_count']} ({summary['no_grasp_ratio']:.4f})"
    )
    print(
        "late_first_grasp_gt_100: "
        f"{summary['late_first_grasp_count']} ({summary['late_first_grasp_ratio']:.4f})"
    )
    print(
        "episodes_with_release: "
        f"{summary['episodes_with_release_count']}"
    )
    print(
        "released_before_target_among_release_episodes: "
        f"{summary['released_before_target_count']} ({summary['released_before_target_ratio']:.4f})"
    )


def print_by_success(summary):
    print_section("By Success")
    for success_flag in [True, False]:
        group_name = "success=True" if success_flag else "success=False"
        stats = summary[success_flag]
        print(group_name)
        print(f"  count: {stats['count']}")
        print(f"  avg_step: {stats['avg_step']:.2f}")
        print(f"  avg_first_grasp_step: {stats['avg_first_grasp_step']:.2f}")
        print(f"  episodes_with_grasp: {stats['episodes_with_grasp_count']}")
        print(
            "  avg_grasp_count_among_grasp_episodes: "
            f"{stats['avg_grasp_count_among_grasp_episodes']:.2f}"
        )
        print(f"  no_grasp_ratio: {stats['no_grasp_ratio']:.4f}")
        print(f"  episodes_with_release: {stats['episodes_with_release_count']}")
        print(
            "  released_before_target_ratio_among_release_episodes: "
            f"{stats['released_before_target_ratio']:.4f}"
        )
        print(f"  avg_potential_delta: {stats['avg_potential_delta']:.4f}")


def print_by_obj_num(summary):
    print_section("By Object Count")
    for obj_num, stats in summary.items():
        print(f"obj_num={obj_num}")
        print(f"  count: {stats['count']}")
        print(f"  success_rate: {stats['success_rate']:.4f}")
        print(f"  avg_arrival_ratio: {stats['avg_arrival_ratio']:.4f}")
        print(f"  no_grasp_ratio: {stats['no_grasp_ratio']:.4f}")
        print(f"  avg_first_grasp_step: {stats['avg_first_grasp_step']:.2f}")
        print(f"  avg_potential_delta: {stats['avg_potential_delta']:.4f}")


def print_release_summary(summary):
    print_section("Release Quality")
    print(f"total_releases: {summary['total_releases']}")
    print(f"release_on_target_ratio: {summary['release_on_target_ratio']:.4f}")
    print(f"release_farther_ratio: {summary['release_farther_ratio']:.4f}")
    print(f"release_closer_ratio: {summary['release_closer_ratio']:.4f}")
    print(f"release_unchanged_ratio: {summary['release_unchanged_ratio']:.4f}")
    print(f"short_hold_ratio_le_5: {summary['short_hold_ratio_le_5']:.4f}")
    print(f"avg_hold_steps: {summary['avg_hold_steps']:.2f}")


def print_scene_categories(summary):
    print_section("Scene Categories")
    print(
        "no_navigation_success_false_min_distance_gt_1_5: "
        f"{summary['no_navigation_count']} ({summary['no_navigation_ratio']:.4f})"
    )
    print(
        "near_object_no_grasp_success_false_min_distance_lt_1_5: "
        f"{summary['near_but_no_grasp_count']} ({summary['near_but_no_grasp_ratio']:.4f})"
    )
    print(
        "grasped_but_not_success: "
        f"{summary['grasped_but_not_success_count']} ({summary['grasped_but_not_success_ratio']:.4f})"
    )
    print(
        "grasped_and_success: "
        f"{summary['grasped_and_success_count']} ({summary['grasped_and_success_ratio']:.4f})"
    )
    print(f"categorized_count: {summary['categorized_count']}")
    print(f"uncategorized_count: {summary['uncategorized_count']}")


def analyze_log(log_path):
    rows = parse_eval_log(log_path)
    if not rows:
        raise ValueError(f"No valid eval_dict entries found in: {log_path}")

    print(f"log_path: {log_path}")
    print_overall(summarize_overall(rows))
    print_by_success(summarize_by_success(rows))
    print_by_obj_num(summarize_by_obj_num(rows))
    print_release_summary(summarize_release_events(rows))
    print_scene_categories(summarize_scene_categories(rows))


def main():
    parser = argparse.ArgumentParser(description="Analyze eval_model.py log metrics.")
    parser.add_argument(
        "--log-path",
        type=str,
        default=DEFAULT_LOG_PATH,
        help="Path to the log file generated from eval_model.py",
    )
    args = parser.parse_args()
    analyze_log(args.log_path)


if __name__ == "__main__":
    main()
