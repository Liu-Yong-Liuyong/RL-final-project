import json
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

env_yaml = "configs/gridworld_easy.yaml"

# exploration_log = "exploration_logs/none_20x20_exploration_obstacles_log.json"
# output_image = "exploration_pictures/gridworld/none_20x20_gridworld_obstacles_firstvisit"
# exploration_log = "exploration_logs/icm_20x20_exploration_obstacles_log.json"
# output_image = "exploration_pictures/gridworld/icm_20x20_gridworld_obstacles_firstvisit"
# exploration_log = "exploration_logs/rnd_20x20_exploration_obstacles_log.json"
# output_image = "exploration_pictures/gridworld/rnd_20x20_gridworld_obstacles_firstvisit"

exploration_log = "exploration_logs/none_50x50_exploration_obstacles_log.json"
output_image = "exploration_pictures/gridworld/none_50x50_gridworld_obstacles_firstvisit"
# exploration_log = "exploration_logs/icm_50x50_exploration_obstacles_log.json"
# output_image = "exploration_pictures/gridworld/icm_50x50_gridworld_obstacles_firstvisit"
# exploration_log = "exploration_logs/rnd_50x50_exploration_obstacles_log.json"
# output_image = "exploration_pictures/gridworld/rnd_50x50_gridworld_obstacles_firstvisit"

# exploration_log = "exploration_logs/none_50x50_exploration_log.json"
# output_image = "exploration_pictures/gridworld/none_50x50_gridworld_firstvisit"
# exploration_log = "exploration_logs/icm_50x50_exploration_log.json"
# output_image = "exploration_pictures/gridworld/icm_50x50_gridworld_firstvisit"
# exploration_log = "exploration_logs/rnd_50x50_exploration_log.json"
# output_image = "exploration_pictures/gridworld/rnd_50x50_gridworld_firstvisit"

# ==========================================
# 1. 定義輔助函數 (Obstacle Generation)
# ==========================================
def get_random_pos(width, height, exclude_set):
    while True:
        pos = (
            random.randint(0, width - 1),
            random.randint(0, height - 1)
        )
        if pos not in exclude_set:
            return pos

def generate_obstacles(width, height, start_pos, goal_pos, num_obstacles, random_seed):
    random.seed(random_seed)
    obstacles = set()
    # 將 start 與 goal 加入排除名單
    exclude = {tuple(start_pos), tuple(goal_pos)}
    
    while len(obstacles) < num_obstacles:
        new_obs = get_random_pos(width, height, exclude)
        obstacles.add(new_obs)
        exclude.add(new_obs)
        
    return obstacles

# ==========================================
# 2. 主程式：讀取資料與還原環境
# ==========================================
def main():
    # --- 讀取 YAML 設定檔 ---
    with open(env_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    
    env_cfg = cfg["env"]["kwargs"]
    width = env_cfg["width"]
    height = env_cfg["height"]
    start_pos = env_cfg["start_pos"]
    goal_pos = env_cfg["goal_pos"]
    num_obstacles = env_cfg["num_obstacles"]
    random_seed = env_cfg["random_seed"]

    # 100% 重建當時的障礙物位置
    obstacles = generate_obstacles(
        width, height, start_pos, goal_pos, num_obstacles, random_seed
    )

    # --- 讀取 Exploration Logs ---
    with open(exploration_log, "r") as f:
        logs = json.load(f)

    # ==========================================
    # 3. 建立探索時間圖 (Heatmap)
    # ==========================================
    # 初始化全為 NaN 的地圖 (代表 never explored)
    heatmap = np.full((height, width), np.nan)
    
    # 填入探索紀錄 (使用 global_step)
    for item in logs:
        x, y = item["coverage_id"]
        # 只記錄「第一次」造訪的 global_step
        if np.isnan(heatmap[y, x]):
            heatmap[y, x] = item["global_step"]
            
    # 標記 obstacle，用特殊值 -1 表示
    OBSTACLE = -1
    for x, y in obstacles:
        heatmap[y, x] = OBSTACLE

    # ==========================================
    # 4. 畫圖 (Visualization)
    # ==========================================
    # 只對非 obstacle 的部分做顏色映射 (Masking)
    masked = np.ma.masked_where(heatmap == OBSTACLE, heatmap)
    
    # 設定 colormap，將 NaN (未探索) 設為黑色
    cmap = plt.cm.plasma.copy()
    cmap.set_bad("black")
    
    plt.figure(figsize=(10, 10))
    
    # 第一層：繪製熱力圖與黑色未探索區域
    im = plt.imshow(masked, cmap=cmap, origin="upper")
    
    # 第二層：精準繪製 Obstacles (使用 RGBA 矩陣，完美填滿網格無空隙)
    # 建立一個與地圖同大小的全透明 RGBA 圖層 (4 個 channel: R, G, B, Alpha)
    obs_layer = np.zeros((height, width, 4)) 
    obs_y, obs_x = np.where(heatmap == OBSTACLE)
    # 將有障礙物的位置設為純白，且完全不透明 (R=1, G=1, B=1, Alpha=1)
    obs_layer[obs_y, obs_x] = [1.0, 1.0, 1.0, 1.0] 
    plt.imshow(obs_layer, origin="upper")
    
    # 繪製 Start & Goal (這兩個用 scatter 沒問題，因為標記不用填滿網格)
    start_x, start_y = start_pos
    goal_x, goal_y = goal_pos
    plt.scatter(start_x, start_y, c="red", marker="o", s=250, label="Start")
    plt.scatter(goal_x, goal_y, c="cyan", marker="o", s=250, label="Goal")
    
    # 圖表收尾設定
    # 由於 imshow 不會自動產生圖例，我們手動建立一個白色方塊的圖例
    obs_patch = mpatches.Patch(color='white', label='Obstacle', ec="black") # ec="black" 加個黑邊框讓白色方塊在圖例中更明顯
    
    # 取得現有的 handles 和 labels (Start 和 Goal)
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # 把 Obstacle 手動加進去圖例清單的最前面
    handles.insert(0, obs_patch)
    labels.insert(0, "Obstacle")
    
    # 放置圖例到最下方
    plt.legend(
        handles=handles,
        labels=labels,
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.05), 
        ncol=3, 
        frameon=True
    )
    plt.colorbar(im, label="First Visit Order (Global Step)")
    plt.title("Exploration Heatmap")
    
    plt.tight_layout()
    
    # 如果你要存檔，可以用這行 (取代或放在 plt.show() 之前)
    plt.savefig(f"{output_image}.png", dpi=300, bbox_inches="tight")
    
    plt.show()

if __name__ == "__main__":
    main()