import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

# --- 設定檔案路徑 ---
exploration_log = "exploration_logs/rnd_button_exploration_log.json"

output_image_first_visit = "exploration_pictures/button/rnd_button_first_visit_mean"
output_image_visit_count = "exploration_pictures/button/rnd_button_visit_count_mean"

def main():
    # ==========================================
    # 1. 讀取資料與數據處理
    # ==========================================
    with open(exploration_log, "r") as f:
        logs = json.load(f)
        
    first_visit_map = {}
    visit_count_map = {}
    target_positions = set()

    for item in logs:
        cid = tuple(item["coverage_id"])
        step = item["global_step"]
        target = tuple(item["target_pos"])
        
        target_positions.add(target)
        
        if cid not in first_visit_map:
            first_visit_map[cid] = step
        else:
            first_visit_map[cid] = min(first_visit_map[cid], step)
            
        visit_count_map[cid] = visit_count_map.get(cid, 0) + 1

    t_x = [t[0] for t in target_positions]
    t_y = [t[1] for t in target_positions]
    t_z = [t[2] for t in target_positions]

    # --- 定義 Voxel 網格參數 ---
    res = 0.1
    x_min, x_max = -0.6, 0.6
    y_min, y_max = 0.3, 1.1
    z_min, z_max = 0.0, 0.5

    nx = int(np.round((x_max - x_min) / res)) + 1
    ny = int(np.round((y_max - y_min) / res)) + 1
    nz = int(np.round((z_max - z_min) / res)) + 1

    x_edges = np.linspace(x_min - res/2, x_max + res/2, nx + 1)
    y_edges = np.linspace(y_min - res/2, y_max + res/2, ny + 1)
    z_edges = np.linspace(z_min - res/2, z_max + res/2, nz + 1)
    x_grid, y_grid, z_grid = np.meshgrid(x_edges, y_edges, z_edges, indexing='ij')

    # 🔥 計算真實的空間比例 (用來鎖定 3D 視角不變形)
    box_aspect = [x_max - x_min, y_max - y_min, z_max - z_min]

    # ==========================================
    # 2. 繪製 First Visit (3D Voxels & 2D 投影)
    # ==========================================
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    filled_fv = np.zeros((nx, ny, nz), dtype=bool)
    val_fv_3d = np.full((nx, ny, nz), np.nan)

    for (x, y, z), step in first_visit_map.items():
        ix = int(np.round((x - x_min) / res))
        iy = int(np.round((y - y_min) / res))
        iz = int(np.round((z - z_min) / res))
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            filled_fv[ix, iy, iz] = True
            val_fv_3d[ix, iy, iz] = step

    valid_fv = [v for v in first_visit_map.values()]
    norm_fv = plt.Normalize(vmin=min(valid_fv), vmax=max(valid_fv))
    cmap_fv_3d = plt.cm.plasma
    colors_fv = np.zeros((nx, ny, nz, 4))
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if filled_fv[ix, iy, iz]:
                    rgba = list(cmap_fv_3d(norm_fv(val_fv_3d[ix, iy, iz])))
                    rgba[3] = 0.3  # 3D 透明度
                    colors_fv[ix, iy, iz] = rgba

    ax1.voxels(x_grid, y_grid, z_grid, filled_fv, facecolors=colors_fv, edgecolor=(0, 0, 0, 0.2), linewidth=0.5)
    ax1.scatter(t_x, t_y, t_z, c='red', marker='*', s=150, edgecolor='black', zorder=10)

    ax1.set_xlim([-0.6, 0.6]) 
    ax1.set_ylim([0.3, 1.1])  
    ax1.set_zlim([0.0, 0.5])
    
    # 🔥 強制 3D 空間依照實際長度比例繪製
    ax1.set_box_aspect(box_aspect)
    
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')
    ax1.set_title("MetaWorld - First Visit Order (3D)")
    
    sm1 = plt.cm.ScalarMappable(cmap=cmap_fv_3d, norm=norm_fv)
    sm1.set_array([])
    fig1.colorbar(sm1, ax=ax1, label="Global Step")
    
    # plt.tight_layout()
    # ax1.view_init(elev=30, azim=-60)
    # plt.savefig(f"{output_image_first_visit}_front.png", dpi=300, bbox_inches="tight")
    # ax1.view_init(elev=-30, azim=120)
    # plt.savefig(f"{output_image_first_visit}_back.png", dpi=300, bbox_inches="tight")
    # print(f"Saved 3D First Visit plots.")

    # --- 繪製 First Visit 2D 投影圖 ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        proj_fv_xy = np.nanmean(val_fv_3d, axis=2)
        proj_fv_xz = np.nanmean(val_fv_3d, axis=1)
        proj_fv_yz = np.nanmean(val_fv_3d, axis=0)

    x_span = x_max - x_min
    y_span = y_max - y_min
    z_span = z_max - z_min

    # --- 針對 First Visit 的 2D 圖 ---
    fig1_2d, axes1 = plt.subplots(
        1, 3, figsize=(20, 5),
        gridspec_kw={'width_ratios': [x_span, x_span, y_span]}
    )
    cmap_fv_2d = plt.cm.plasma.copy()
    cmap_fv_2d.set_bad('black')

    ext_xy = [x_min - res/2, x_max + res/2, y_min - res/2, y_max + res/2]
    ext_xz = [x_min - res/2, x_max + res/2, z_min - res/2, z_max + res/2]
    ext_yz = [y_min - res/2, y_max + res/2, z_min - res/2, z_max + res/2]

    # 🔥 加入 aspect='equal' 確保 2D 投影的長寬物理刻度一致
    im1_0 = axes1[0].imshow(proj_fv_xy.T, origin='lower', extent=ext_xy, cmap=cmap_fv_2d, norm=norm_fv, aspect='equal')
    axes1[0].set_title('XY (Top-Down)')
    
    im1_1 = axes1[1].imshow(proj_fv_xz.T, origin='lower', extent=ext_xz, cmap=cmap_fv_2d, norm=norm_fv, aspect='equal')
    axes1[1].set_title('XZ (Front)')
    
    im1_2 = axes1[2].imshow(proj_fv_yz.T, origin='lower', extent=ext_yz, cmap=cmap_fv_2d, norm=norm_fv, aspect='equal')
    axes1[2].set_title('YZ (Side)')

    for tx, ty, tz in target_positions:
        axes1[0].scatter(tx, ty, c='red', marker='*', s=150, edgecolor='black')
        axes1[1].scatter(tx, tz, c='red', marker='*', s=150, edgecolor='black')
        axes1[2].scatter(ty, tz, c='red', marker='*', s=150, edgecolor='black')

    fig1_2d.colorbar(im1_2, ax=axes1.ravel().tolist(), label="Global Step")
    fig1_2d.suptitle("MetaWorld - First Visit Order (2D Projections)")
    plt.savefig(f"{output_image_first_visit}_2d_proj.png", dpi=300, bbox_inches="tight")
    print(f"Saved 2D First Visit projections.")


    # ==========================================
    # 3. 繪製 Visit Count (3D Voxels & 2D 投影)
    # ==========================================
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    filled_vc = np.zeros((nx, ny, nz), dtype=bool)
    val_vc_3d = np.full((nx, ny, nz), np.nan)

    for (x, y, z), count in visit_count_map.items():
        ix = int(np.round((x - x_min) / res))
        iy = int(np.round((y - y_min) / res))
        iz = int(np.round((z - z_min) / res))
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            filled_vc[ix, iy, iz] = True
            val_vc_3d[ix, iy, iz] = count

    valid_vc = [v for v in visit_count_map.values()]
    norm_vc = plt.Normalize(vmin=min(valid_vc), vmax=max(valid_vc))
    cmap_vc_3d = plt.cm.viridis
    colors_vc = np.zeros((nx, ny, nz, 4))
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if filled_vc[ix, iy, iz]:
                    rgba = list(cmap_vc_3d(norm_vc(val_vc_3d[ix, iy, iz])))
                    rgba[3] = 0.3  # 3D 透明度
                    colors_vc[ix, iy, iz] = rgba
                    
    ax2.voxels(x_grid, y_grid, z_grid, filled_vc, facecolors=colors_vc, edgecolor=(0, 0, 0, 0.2), linewidth=0.5)
    ax2.scatter(t_x, t_y, t_z, c='red', marker='*', s=150, edgecolor='black', zorder=10)

    ax2.set_xlim([-0.6, 0.6])
    ax2.set_ylim([0.3, 1.1])
    ax2.set_zlim([0.0, 0.5])
    
    # 🔥 強制 3D 空間依照實際長度比例繪製
    ax2.set_box_aspect(box_aspect)
    
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')
    ax2.set_title("MetaWorld - Visit Count (3D)")
    
    sm2 = plt.cm.ScalarMappable(cmap=cmap_vc_3d, norm=norm_vc)
    sm2.set_array([])
    fig2.colorbar(sm2, ax=ax2, label="Visit Count")
    
    # plt.tight_layout()
    # ax2.view_init(elev=30, azim=-60)
    # plt.savefig(f"{output_image_visit_count}_front.png", dpi=300, bbox_inches="tight")
    # ax2.view_init(elev=-30, azim=120)
    # plt.savefig(f"{output_image_visit_count}_back.png", dpi=300, bbox_inches="tight")
    # print(f"Saved 3D Visit Count plots.")

    # --- 繪製 Visit Count 2D 投影圖 ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        proj_vc_xy = np.nanmean(val_vc_3d, axis=2)
        proj_vc_xz = np.nanmean(val_vc_3d, axis=1)
        proj_vc_yz = np.nanmean(val_vc_3d, axis=0)

    fig2_2d, axes2 = plt.subplots(
        1, 3, figsize=(20, 5),
        gridspec_kw={'width_ratios': [x_span, x_span, y_span]}
    )
    cmap_vc_2d = plt.cm.viridis.copy()
    cmap_vc_2d.set_bad('black')

    # 🔥 加入 aspect='equal'
    im2_0 = axes2[0].imshow(proj_vc_xy.T, origin='lower', extent=ext_xy, cmap=cmap_vc_2d, norm=norm_vc, aspect='equal')
    axes2[0].set_title('XY (Top-Down)')
    
    im2_1 = axes2[1].imshow(proj_vc_xz.T, origin='lower', extent=ext_xz, cmap=cmap_vc_2d, norm=norm_vc, aspect='equal')
    axes2[1].set_title('XZ (Front)')
    
    im2_2 = axes2[2].imshow(proj_vc_yz.T, origin='lower', extent=ext_yz, cmap=cmap_vc_2d, norm=norm_vc, aspect='equal')
    axes2[2].set_title('YZ (Side)')

    for tx, ty, tz in target_positions:
        axes2[0].scatter(tx, ty, c='red', marker='*', s=150, edgecolor='black')
        axes2[1].scatter(tx, tz, c='red', marker='*', s=150, edgecolor='black')
        axes2[2].scatter(ty, tz, c='red', marker='*', s=150, edgecolor='black')

    fig2_2d.colorbar(im2_2, ax=axes2.ravel().tolist(), label="Visit Count")
    fig2_2d.suptitle("MetaWorld - Visit Count (2D Projections)")
    plt.savefig(f"{output_image_visit_count}_2d_proj.png", dpi=300, bbox_inches="tight")
    print(f"Saved 2D Visit Count projections.")

    # 還原視角並顯示
    ax1.view_init(elev=30, azim=-60)
    ax2.view_init(elev=30, azim=-60)
    # plt.show()

if __name__ == "__main__":
    main()