'''
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def plot_visit_heatmap_from_json(
    json_path,
    dims=(0, 1),   # (0,1) = x-y, (0,2) = x-z
    title="ICM Visit Count Heatmap"
):
    with open(json_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    points = []
    for item in logs:
        cid = item.get("coverage_id", None)
        if cid is None or len(cid) < max(dims) + 1:
            continue
        points.append((cid[dims[0]], cid[dims[1]]))

    visits = Counter(points)

    xs = sorted(set(x for x, y in visits.keys()))
    ys = sorted(set(y for x, y in visits.keys()))

    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: i for i, y in enumerate(ys)}

    heatmap = np.zeros((len(ys), len(xs)))

    for (x, y), count in visits.items():
        heatmap[y_to_idx[y], x_to_idx[x]] = count

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, origin="lower", aspect="auto")
    plt.colorbar(label="Visit Count")
    plt.xticks(range(len(xs)), xs, rotation=45)
    plt.yticks(range(len(ys)), ys)
    plt.xlabel(f"coverage_id[{dims[0]}]")
    plt.ylabel(f"coverage_id[{dims[1]}]")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    plot_visit_heatmap_from_json(
        "exploration_logs/icm_exploration_log.json",
        dims=(0, 1),
        title="ICM Reach Exploration (x-y)"
        #dims=(0, 2),
        #title="ICM Reach Exploration (x-z)"
    )
if __name__ == "__main__":
    main()

'''

import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def plot_visit_heatmap_from_json(
    json_path,
    dims=(0, 1),   # (0,1) = x-y, (0,2) = x-z
    title="ICM Visit Count Heatmap",
    use_target_from_json=False,
):
    with open(json_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    points = []
    target_points = []

    for item in logs:
        cid = item.get("coverage_id", None)
        if cid is None or len(cid) < max(dims) + 1:
            continue
        points.append((cid[dims[0]], cid[dims[1]]))

        if use_target_from_json:
            target = item.get("target_pos", None)
            if target is not None and len(target) >= max(dims) + 1:
                tx = round(float(target[dims[0]]), 1)
                ty = round(float(target[dims[1]]), 1)
                target_points.append((tx, ty))

    visits = Counter(points)

    xs = sorted(set(x for x, y in visits.keys()))
    ys = sorted(set(y for x, y in visits.keys()))

    # 如果要畫 target，也把 target 的 bin 納入座標軸
    if use_target_from_json and len(target_points) > 0:
        xs = sorted(set(xs) | set(x for x, y in target_points))
        ys = sorted(set(ys) | set(y for x, y in target_points))

    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: i for i, y in enumerate(ys)}

    heatmap = np.zeros((len(ys), len(xs)))

    for (x, y), count in visits.items():
        heatmap[y_to_idx[y], x_to_idx[x]] = count

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, origin="lower", aspect="auto")
    plt.colorbar(label="Visit Count")
    plt.xticks(range(len(xs)), xs, rotation=45)
    plt.yticks(range(len(ys)), ys)
    plt.xlabel(f"coverage_id[{dims[0]}]")
    plt.ylabel(f"coverage_id[{dims[1]}]")
    plt.title(title)

    if use_target_from_json and len(target_points) > 0:
        # 若 target 幾乎固定，取平均後 round 到同樣 bin
        mean_tx = round(np.mean([p[0] for p in target_points]), 1)
        mean_ty = round(np.mean([p[1] for p in target_points]), 1)

        if mean_tx in x_to_idx and mean_ty in y_to_idx:
            plt.scatter(
                x_to_idx[mean_tx],
                y_to_idx[mean_ty],
                marker="x",
                s=140,
                linewidths=2,
                label="Target (avg from json)"
            )
            plt.legend()

    #plt.tight_layout()
    #plt.show()
    '''
    plt.tight_layout()
    plt.savefig("rnd_dc0_reach_heatmap_yz.png", dpi=200)
    '''
    save_dir = "exploration_pictures"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "rnd_dc0_reach_heatmap_yz.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"saved to {save_path}")

def main():
    '''
    plot_visit_heatmap_from_json(
        "exploration_logs/icm_exploration_log.json",
        dims=(0, 1),
        title="ICM Reach Exploration (x-y)",
        use_target_from_json=True,
    )
    plot_visit_heatmap_from_json(
        "exploration_logs/rnd_exploration_log.json",
        dims=(1, 2),
        title="RND Reach Exploration (y-z)",
        use_target_from_json=True,
    )
    
    plot_visit_heatmap_from_json(
        "exploration_logs/none_exploration_log.json",
        dims=(1, 2),
        title="ONLY PPO Reach Exploration (y-z)",
        use_target_from_json=True,
    )
    
    plot_visit_heatmap_from_json(
        "exploration_logs/none_dc0_exploration_log.json",
        dims=(1, 2),
        title="ONLY PPO Reach Exploration (y-z)",
        use_target_from_json=True,
    )
    
    plot_visit_heatmap_from_json(
        "exploration_logs/icm_dc0_exploration_log.json",
        dims=(1, 2),
        title="ICM Reach Exploration (y-z)",
        use_target_from_json=True,
    )
    '''
    plot_visit_heatmap_from_json(
        "exploration_logs/rnd_dc0_exploration_log.json",
        dims=(1, 2),
        title="RND Reach Exploration (y-z)",
        use_target_from_json=True,
    )    
if __name__ == "__main__":
    main()