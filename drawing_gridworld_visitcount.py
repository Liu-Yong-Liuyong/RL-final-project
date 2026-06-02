import json
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

env_yaml = "configs/gridworld_easy.yaml"

# exploration_log = "exploration_logs/none_20x20_exploration_obstacles_log.json"
# output_image = "exploration_pictures/gridworld/none_20x20_gridworld_obstacles_visitcount"
# exploration_log = "exploration_logs/icm_20x20_exploration_obstacles_log.json"
# output_image = "exploration_pictures/gridworld/icm_20x20_gridworld_obstacles_visitcount" 
# exploration_log = "exploration_logs/rnd_20x20_exploration_obstacles_log.json"
# output_image = "exploration_pictures/gridworld/rnd_20x20_gridworld_obstacles_visitcount" 

exploration_log = "exploration_logs/none_50x50_exploration_obstacles_log.json"
output_image = "exploration_pictures/gridworld/none_50x50_gridworld_obstacles_visitcount"
# exploration_log = "exploration_logs/icm_50x50_exploration_obstacles_log.json"
# output_image = "exploration_pictures/gridworld/icm_50x50_gridworld_obstacles_visitcount" 
# exploration_log = "exploration_logs/rnd_50x50_exploration_obstacles_log.json"
# output_image = "exploration_pictures/gridworld/rnd_50x50_gridworld_obstacles_visitcount" 

# exploration_log = "exploration_logs/none_50x50_exploration_log.json"
# output_image = "exploration_pictures/gridworld/none_50x50_gridworld_visitcount"
# exploration_log = "exploration_logs/icm_50x50_exploration_log.json"
# output_image = "exploration_pictures/gridworld/icm_50x50_gridworld_visitcount"
# exploration_log = "exploration_logs/rnd_50x50_exploration_log.json"
# output_image = "exploration_pictures/gridworld/rnd_50x50_gridworld_visitcount"

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
    # 3. 建立探索次數圖 (Visit Count Heatmap)
    # ==========================================
    # 初始化一個純粹用來計算「造訪次數」的地圖
    visit_counts = np.zeros((height, width))
    
    # 填入探索紀錄 (累加次數)
    for item in logs:
        x, y = item["coverage_id"]
        visit_counts[y, x] += 1
            
    # 建立一個獨立的 Obstacle Mask (True 代表是障礙物)
    obs_mask = np.zeros((height, width), dtype=bool)
    for x, y in obstacles:
        obs_mask[y, x] = True

    # ==========================================
    # 4. 畫圖 (Visualization)
    # ==========================================
    # 只對非 obstacle 且有造訪過的部分做顏色映射 (Masking)
    # 條件：如果是障礙物 (obs_mask) 或 次數為 0 (visit_counts == 0) 就遮蔽掉
    masked = np.ma.masked_where(obs_mask | (visit_counts == 0), visit_counts)
    
    # 設定 colormap，將被遮蔽 (未探索) 的地方設為黑色
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("black")
    
    plt.figure(figsize=(10, 10))
    
    # 第一層：繪製熱力圖與黑色未探索區域
    im = plt.imshow(masked, cmap=cmap, origin="upper")
    
    # 第二層：精準繪製 Obstacles
    # 建立一個與地圖同大小的全透明 RGBA 圖層
    obs_layer = np.zeros((height, width, 4)) 
    # 直接利用剛剛建立的 bool mask，把障礙物的位置設為純白不透明
    obs_layer[obs_mask] = [1.0, 1.0, 1.0, 1.0] 
    plt.imshow(obs_layer, origin="upper")
    
    # 繪製 Start & Goal
    start_x, start_y = start_pos
    goal_x, goal_y = goal_pos
    plt.scatter(start_x, start_y, c="red", marker="o", s=250, label="Start")
    plt.scatter(goal_x, goal_y, c="cyan", marker="o", s=250, label="Goal")
    
    # 圖表收尾設定
    obs_patch = mpatches.Patch(color='white', label='Obstacle', ec="black") 
    
    handles, labels = plt.gca().get_legend_handles_labels()
    
    handles.insert(0, obs_patch)
    labels.insert(0, "Obstacle")
    
    plt.legend(
        handles=handles,
        labels=labels,
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.05), 
        ncol=3, 
        frameon=True
    )
    plt.colorbar(im, label="Visit Count (Frequency)")
    plt.title("Exploration Visit Count Heatmap")
    
    plt.tight_layout()
    
    # 如果你要存檔，可以用這行
    plt.savefig(f"{output_image}.png", dpi=300, bbox_inches="tight")
    
    plt.show()

if __name__ == "__main__":
    main()