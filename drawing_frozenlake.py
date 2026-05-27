'''
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_frozenlake_visit_heatmap(json_path, nrow=8, ncol=8, title="FrozenLake Visit Heatmap"):
    with open(json_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    heatmap = np.zeros((nrow, ncol), dtype=np.float32)

    for item in logs:
        state = item.get("coverage_id", None)
        if state is None:
            continue

        state = int(state)
        row = state // ncol
        col = state % ncol

        if 0 <= row < nrow and 0 <= col < ncol:
            heatmap[row, col] += 1

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, origin="upper", aspect="equal")
    plt.colorbar(label="Visit Count")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.title(title)

    plt.xticks(range(ncol))
    plt.yticks(range(nrow))
    plt.tight_layout()
    plt.show()

def main():
    plot_frozenlake_visit_heatmap(
        "exploration_logs/icm_exploration_log.json",
        nrow=8,
        ncol=8,
        title="ICM FrozenLake Visit Count"
    )
if __name__ == "__main__":
    main()
'''
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_frozenlake_visit_heatmap(
    json_path,
    nrow=8,
    ncol=8,
    title="FrozenLake Visit Heatmap",
    use_goal_from_json=True,
):
    with open(json_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    heatmap = np.zeros((nrow, ncol), dtype=np.float32)
    goal_rc = None

    for item in logs:
        cid = item.get("coverage_id", None)
        if cid is None:
            continue

        # 支援兩種格式：
        # 1. coverage_id = state index
        # 2. coverage_id = [row, col]
        if isinstance(cid, (list, tuple)):
            if len(cid) < 2:
                continue
            row, col = int(cid[0]), int(cid[1])
        else:
            state = int(cid)
            row = state // ncol
            col = state % ncol

        if 0 <= row < nrow and 0 <= col < ncol:
            heatmap[row, col] += 1

        if use_goal_from_json and goal_rc is None:
            if "goal_rc" in item and item["goal_rc"] is not None:
                gr, gc = item["goal_rc"]
                goal_rc = (int(gr), int(gc))
            elif "goal_state" in item and item["goal_state"] is not None:
                goal_state = int(item["goal_state"])
                goal_rc = (goal_state // ncol, goal_state % ncol)

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, origin="upper", aspect="equal")
    plt.colorbar(label="Visit Count")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.title(title)

    plt.xticks(range(ncol))
    plt.yticks(range(nrow))

    if goal_rc is not None:
        gr, gc = goal_rc
        plt.scatter(
            gc, gr,
            marker="x",
            s=160,
            linewidths=2,
            label="Goal"
        )
        plt.legend()

    #plt.tight_layout()
    #plt.show()
    plt.tight_layout()
    plt.savefig("none_frozenlake_heatmap.png", dpi=200)
    print("saved to frozenlake_heatmap.png")


def main():
    '''
    plot_frozenlake_visit_heatmap(
        "exploration_logs/icm_exploration_frozenlake_log.json",
        nrow=8,
        ncol=8,
        title="ICM FrozenLake Visit Count",
        use_goal_from_json=True,
    )
    plot_frozenlake_visit_heatmap(
        "exploration_logs/rnd_exploration_frozenlake_log.json",
        nrow=8,
        ncol=8,
        title="RND FrozenLake Visit Count",
        use_goal_from_json=True,
    )
    '''
    plot_frozenlake_visit_heatmap(
        "exploration_logs/none_exploration_frozenlake_log.json",
        nrow=8,
        ncol=8,
        title="ONLY PPO FrozenLake Visit Count",
        use_goal_from_json=True,
    )

if __name__ == "__main__":
    main()