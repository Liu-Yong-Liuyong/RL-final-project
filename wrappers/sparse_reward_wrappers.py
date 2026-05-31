#for carpole and robot arms...
import gymnasium as gym
import numpy as np


class SparseRewardWrapper(gym.Wrapper):
    def __init__(self, env, success_reward=10.0, failure_reward=0.0, dense_scale=0.0):
        super().__init__(env)
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.dense_scale = dense_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        #sparse_reward = self.success_reward if info.get("success", False) else self.failure_reward
        # sparse_reward = self.success_reward if info.get("success", False) else 0.01 * reward
        if info.get("success", False):
            sparse_reward = self.success_reward
        else:
            sparse_reward = self.failure_reward + (self.dense_scale * reward)
        info["original_reward"] = reward

        return obs, sparse_reward, terminated, truncated, info



# good at normal ppo, icm, rnd for reach-v3
import gymnasium as gym
import numpy as np


class GoalThresholdRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        success_reward=50.0,
        threshold_rewards=None,
        failure_reward=0.0,
        use_dense_fallback=False,
        dense_coef=0.0,
    ):
        super().__init__(env)
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.use_dense_fallback = use_dense_fallback
        self.dense_coef = dense_coef

        # 每個元素: (distance_threshold, reward)
        if threshold_rewards is None:
            threshold_rewards = [
                (0.12, 0.05),
                (0.07, 0.1),
                (0.03, 0.5),
            ]

        # 由大到小排序，方便依序判斷
        self.threshold_rewards = sorted(threshold_rewards, key=lambda x: x[0], reverse=True)

        # 記錄本 episode 已經領過哪些 threshold reward
        self.threshold_hit = None
        self.success_given = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 每個 episode 重設
        self.threshold_hit = {thr: False for thr, _ in self.threshold_rewards}
        self.success_given = False

        return obs, info

    def _extract_positions(self, obs):
        obs = np.asarray(obs)

        # 你需要依 MetaWorld observation 結構確認 target_pos index
        # 這裡先沿用你原本的假設寫法
        eef_pos = obs[:3]
        target_pos = obs[36:39]

        return eef_pos, target_pos

    def step(self, action):
        obs, dense_reward, terminated, truncated, info = self.env.step(action)

        success = bool(info.get("success", False))
        eef_pos, target_pos = self._extract_positions(obs)
        distance = np.linalg.norm(eef_pos - target_pos)

        reward = self.failure_reward
        reward_source = "failure"

        # success reward 每個 episode 只給一次
        if success and not self.success_given:
            reward = self.success_reward
            self.success_given = True
            reward_source = "success"

        else:
            matched = False

            # threshold reward 每層每個 episode 只給一次
            for threshold, threshold_reward in self.threshold_rewards:
                if distance < threshold and not self.threshold_hit[threshold]:
                    reward = threshold_reward
                    self.threshold_hit[threshold] = True
                    matched = True
                    reward_source = f"threshold_{threshold}"
                    break

            # 如果沒有命中任何新 threshold，再決定是否給 dense fallback
            if not matched and reward_source == "failure" and self.use_dense_fallback:
                reward = self.dense_coef * dense_reward
                reward_source = "dense_fallback"

        info["dense_reward"] = float(dense_reward)
        info["goal_distance"] = float(distance)
        info["threshold_reward"] = float(reward)
        info["reward_source"] = reward_source

        return obs, reward, terminated, truncated, info


#trying pick place v3
class PickPlaceMilestoneRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        success_reward=50.0,
        reach_threshold_rewards=None,
        target_threshold_rewards=None,
        grasp_reward=0.5,
        lift_reward=1.0,
        #near_target_reward=2.0,
        #target_threshold=0.07,
        lift_height=0.04,
        failure_reward=0.0,
        use_dense_fallback=False,
        dense_coef=0.0,
    ):
        super().__init__(env)

        self.success_reward = success_reward
        self.grasp_reward = grasp_reward
        self.lift_reward = lift_reward
        #self.near_target_reward = near_target_reward

        #self.target_threshold = target_threshold
        self.lift_height = lift_height

        self.failure_reward = failure_reward
        self.use_dense_fallback = use_dense_fallback
        self.dense_coef = dense_coef

        if reach_threshold_rewards is None:
            reach_threshold_rewards = [
                (0.10, 0.05),
                (0.05, 0.1),
                (0.02, 0.2),
            ]

        # 由大到小排序
        self.reach_threshold_rewards = sorted(
            reach_threshold_rewards,
            key=lambda x: x[0],
            reverse=True
        )
        if target_threshold_rewards is None:
            target_threshold_rewards = [
                (0.20, 0.2),
                (0.10, 0.5),
                (0.05, 1.0),
            ]

        self.target_threshold_rewards = sorted(
            target_threshold_rewards,
            key=lambda x: x[0],
            reverse=True
        )
        self._reset_flags()

    def _reset_flags(self):
        self.grasped = False
        self.lifted = False
        #self.near_target = False
        self.success_given = False
        self.reach_threshold_hit = {thr: False for thr, _ in self.reach_threshold_rewards}
        self.target_threshold_hit = {thr: False for thr, _ in self.target_threshold_rewards}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_flags()
        return obs, info

    def _extract_positions(self, obs):
        obs = np.asarray(obs)

        hand_pos = obs[:3]
        obj_pos = obs[4:7]
        target_pos = np.asarray(self.env.get_target_pos())

        return hand_pos, obj_pos, target_pos

    def step(self, action):
        obs, dense_reward, terminated, truncated, info = self.env.step(action)

        hand_pos, obj_pos, target_pos = self._extract_positions(obs)

        hand_obj_dist = np.linalg.norm(hand_pos - obj_pos)
        obj_target_dist = np.linalg.norm(obj_pos - target_pos)

        #success = bool(info.get("success", False))
        raw_success = bool(info.get("success", False))
        success = raw_success and self.grasped

        #grasp_success = bool(info.get("grasp_success", False))

        # 桌面高度約 0.02，超過桌面一定高度視為 lifted
        obj_height = float(obj_pos[2])
        lifted_enough = obj_height > (0.02 + self.lift_height)

        grasp_success = bool(info.get("grasp_success", False)) or (
            hand_obj_dist < 0.05 and obj_height > 0.025
        )

        reward = self.failure_reward
        reward_source = "failure"
        matched = False
        
        # 1. success：每個 episode 只給一次
        if success and not self.success_given:
            reward = self.success_reward
            self.success_given = True
            reward_source = "success"
            matched = True

        # 2. target 分段 reward：每層每個 episode 只給一次
        else:
            if self.grasped:
                for threshold, threshold_reward in self.target_threshold_rewards:
                    if obj_target_dist < threshold and not self.target_threshold_hit[threshold]:
                        reward = threshold_reward
                        self.target_threshold_hit[threshold] = True
                        reward_source = f"near_target_{threshold}"
                        matched = True
                        break

            # 3. grasp：每個 episode 只給一次
            if not matched and grasp_success and not self.grasped:
                reward = self.grasp_reward
                self.grasped = True
                reward_source = "grasp"
                matched = True

            # 4. lifted：每個 episode 只給一次
            if not matched and lifted_enough and not self.lifted:
                reward = self.lift_reward
                self.lifted = True
                reward_source = "lift"
                matched = True


            # 5. hand-to-object 分段 reward：每層每個 episode 只給一次
            if not matched and not self.grasped and not self.lifted:
                for threshold, threshold_reward in self.reach_threshold_rewards:
                    if hand_obj_dist < threshold and not self.reach_threshold_hit[threshold]:
                        reward = threshold_reward
                        self.reach_threshold_hit[threshold] = True
                        reward_source = f"reach_object_{threshold}"
                        matched = True
                        break
        # 6. 若沒有命中任何 milestone，可選擇保留極弱 dense fallback
        if not matched and self.use_dense_fallback:
            reward = self.dense_coef * dense_reward
            reward_source = "dense_fallback"

        info["raw_success"] = raw_success
        info["success"] = success
        info["dense_reward"] = float(dense_reward)
        info["hand_obj_dist"] = float(hand_obj_dist)
        info["obj_target_dist"] = float(obj_target_dist)
        info["obj_height"] = float(obj_height)
        info["milestone_reward"] = float(reward)
        info["reward_source"] = reward_source
        if reward_source != "dense_fallback" and reward_source != "failure":
            print(
                f"[MILESTONE] reward_source={reward_source}, "
                f"hand_obj_dist={hand_obj_dist:.4f}, "
                f"obj_target_dist={obj_target_dist:.4f}, "
                f"reward={reward:.4f}"
            )
        return obs, reward, terminated, truncated, info