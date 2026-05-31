import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def plot_relative_intrinsic_heatmap(
    json_path,
    cov_dims=(0, 1),
    target_dims=(0, 1),
    title="Relative Average Intrinsic Reward Heatmap"
):
    with open(json_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    reward_sum = defaultdict(float)
    reward_count = defaultdict(int)

    for item in logs:
        cid = item.get("coverage_id", None)
        target = item.get("target_pos", None)
        intrinsic = item.get("intrinsic_reward", None)

        if cid is None or target is None or intrinsic is None:
            continue
        if len(cid) < max(cov_dims) + 1 or len(target) < max(target_dims) + 1:
            continue

        rel_x = round(float(cid[cov_dims[0]]) - round(float(target[target_dims[0]]), 1), 1)
        rel_y = round(float(cid[cov_dims[1]]) - round(float(target[target_dims[1]]), 1), 1)

        key = (rel_x, rel_y)
        reward_sum[key] += float(intrinsic)
        reward_count[key] += 1

    keys = list(reward_sum.keys())
    xs = sorted(set(x for x, y in keys))
    ys = sorted(set(y for x, y in keys))

    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: i for i, y in enumerate(ys)}

    heatmap = np.full((len(ys), len(xs)), np.nan)

    for (x, y), s in reward_sum.items():
        heatmap[y_to_idx[y], x_to_idx[x]] = s / reward_count[(x, y)]

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, origin="lower", aspect="auto")
    plt.colorbar(label="Average Intrinsic Reward")
    plt.xticks(range(len(xs)), xs, rotation=45)
    plt.yticks(range(len(ys)), ys)
    plt.xlabel("relative x")
    plt.ylabel("relative y")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    import json
    import numpy as np

    with open("exploration_logs/icm_exploration_log.json", "r", encoding="utf-8") as f:
        logs = json.load(f)

    vals = [item["intrinsic_reward"] for item in logs if "intrinsic_reward" in item]
    print("count =", len(vals))
    print("min =", np.min(vals))
    print("max =", np.max(vals))
    print("mean =", np.mean(vals))
    print("std =", np.std(vals))
    print("first 20 =", vals[:20])
    plot_relative_intrinsic_heatmap(
        "exploration_logs/icm_exploration_log.json",
        cov_dims=(0, 1),
        target_dims=(0, 1),
        title="ICM Reach Relative Intrinsic Reward (x-y)"
    )
if __name__ == "__main__":
    main()