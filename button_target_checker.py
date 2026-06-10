import json

log_file = "exploration_logs/icm_button_exploration_log.json"

with open(log_file, "r") as f:
    logs = json.load(f)

print("=== Target Position 變化驗證 ===")
current_target = None
change_count = 0

for item in logs:
    step = item["global_step"]
    target = tuple(item["target_pos"])
    
    # 如果 target 改變了，印出當時的 step
    if target != current_target:
        print(f"Step {step:5d}: Target changed to {target}")
        current_target = target
        change_count += 1
        
        # 為了避免洗版，印出前 10 次變化就好
        if change_count > 10:
            print("...")
            break