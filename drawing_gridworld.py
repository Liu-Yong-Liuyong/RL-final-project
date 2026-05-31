import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

def plot_gridworld_heatmap(yaml_path, log_json_path, save_heatmap_path):
    # 1. 讀取 YAML 設定檔取得地圖尺寸、起點與終點
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    env_kwargs = config["env"]["kwargs"]
    width = env_kwargs.get("width", 20)
    height = env_kwargs.get("height", 20)
    start_pos = env_kwargs.get("start_pos", [0, 0])
    goal_pos = env_kwargs.get("goal_pos", [19, 19])
    
    method_name = config["wrappers"]["exploration"]["method"].upper()

    # 2. 讀取軌跡 Log 數據
    with open(log_json_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    # 3. 初始化計數矩陣 (Matplotlib 矩陣索引為 [row, col] -> [y, x])
    heatmap_matrix = np.zeros((height, width))

    for log in logs:
        coord = log.get("coverage_id")
        if coord is not None and len(coord) == 2:
            x, y = int(coord[0]), int(coord[1])
            if 0 <= x < width and 0 <= y < height:
                heatmap_matrix[y, x] += 1

    # 4. 使用 matplotlib.pyplot.imshow 繪製熱圖
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # origin="upper" 讓 y=0 在最上方，往下 y 逐漸增大
    # interpolation="nearest" 確保格子邊界清晰
    im = ax.imshow(heatmap_matrix, cmap="viridis", origin="upper", interpolation="nearest")
    
    # 加上右側顏色條
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Visit Counts")

    # 5. 標註起點 (S) 與終點 (G)
    # 在 origin="upper" 的狀況下，直覺的 (x, y) 位置對應完全一致
    ax.text(start_pos[0], start_pos[1], "S", color="orange", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(goal_pos[0], goal_pos[1], "G", color="red", ha="center", va="center", fontsize=14, weight="bold")

    # 6. 設定座標軸範圍與標籤
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    
    # 若格子太多，每隔 5 格顯示一次刻度
    if width > 10:
        ax.set_xticks(np.arange(0, width, 5))
    if height > 10:
        ax.set_yticks(np.arange(0, height, 5))

    # 將 X 軸刻度移到最上方（如果你希望看起來更像矩陣索引，也可以留著這行；若不需要可直接註解掉）
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.title(f"GridWorld Exploration Trajectory Heatmap ({method_name})", fontsize=14, pad=20)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    plt.tight_layout()
    plt.savefig(save_heatmap_path, dpi=300)
    plt.show()

# 執行繪圖
# plot_gridworld_heatmap("configs/gridworld_easy.yaml", "exploration_logs/icm_dc0_exploration_log.json", "exploration_pictures/gridworld/icm_gridworld_heatmap")
plot_gridworld_heatmap("configs/gridworld_easy.yaml", "exploration_logs/rnd_dc0_exploration_log.json", "exploration_pictures/gridworld/rnd_gridworld_heatmap")