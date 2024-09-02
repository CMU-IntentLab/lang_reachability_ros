import os
import json

import matplotlib.pyplot as plt
import numpy as np


result_roots = ["/home/zli133/shared/ml_projects/backups/lang_reachability_ros/results/rtabmap_mppi_vlm_reachability/2024-08-28-21:06:45",
                "/home/zli133/shared/ml_projects/backups/lang_reachability_ros/results/rtabmap_mppi_vlm_reachability/2024-08-28-21:06:45"]
gt_map_without_constrain_path = "/home/zli133/shared/ml_projects/backups/lang_reachability_ros/maps/rug_2_topdown.npy"
gt_map_with_constrain_path = "/home/zli133/shared/ml_projects/backups/lang_reachability_ros/maps/rug_2_topdown_with_rug.npy"

def load_map_data(result_root):
    map_dict = {}
    maps = np.load(os.path.join(result_root, "semantic_map_over_time.npy"))  # H x W x N
    info = np.load(os.path.join(result_root, "semantic_map_times.npy"))  # N
    print("semantic map shape: ")
    print(maps.shape)
    print(info.shape)
    if maps.ndim == 2:
        maps = maps.reshape(maps.shape[0], maps.shape[1], 1)
    for i, (time, resolution, (origin_x, origin_y)) in enumerate(info):
        # print(time)
        if i >= maps.shape[-1]:
            break
        map_height, map_width = maps[:, :, i].shape
        map_dict[time] = {"data": maps[:, :, i],
                          "resolution": resolution,
                          "origin": (origin_x, origin_y),
                          "shape": (map_height, map_width)}

    return map_dict

def crop_gt_map(occ_map, gt_map_path):
    gt_map = np.load(gt_map_path)
    gt_map = np.copy(gt_map).astype(np.float32)
    gt_map = np.fliplr(gt_map)
    gt_map[gt_map == -1] = 50  # Set unknown to middle gray
    gt_map = gt_map / 100.0  # Normalize to range [0, 1]

    map_height, map_width = occ_map.shape

    origin_x_on_gt = 225 - 58
    origin_y_on_gt = 83 - 16

    # Calculate crop coordinates
    x_start = origin_x_on_gt
    y_start = origin_y_on_gt
    x_end = x_start + map_width
    y_end = y_start + map_height

    # Ensure coordinates are within bounds of gt_map
    x_start = max(x_start, 0)
    y_start = max(y_start, 0)
    x_end = min(x_end, gt_map.shape[1])
    y_end = min(y_end, gt_map.shape[0])

    # Crop the ground truth map
    gt_map = gt_map[y_start:y_end, x_start:x_end]
    return gt_map

def get_gt_semantic_map(result_root):
    semantic_maps_dict = load_map_data(result_root)
    latest_semantic_map = list(semantic_maps_dict.values())[-1]["data"]
    # print(latest_semantic_map.shape)

    gt_map_without_constrain = crop_gt_map(latest_semantic_map, gt_map_without_constrain_path)
    gt_map_with_constrain = crop_gt_map(latest_semantic_map, gt_map_with_constrain_path)

    gt_semantic_map = gt_map_with_constrain - gt_map_without_constrain
    gt_semantic_map[gt_semantic_map < 0] = 0

    # plt.imshow(gt_semantic_map, cmap="gray", origin="lower")
    # plt.title("Difference between Constrained and Unconstrained Maps")
    # plt.show()

    return gt_semantic_map

def compare_vlm_map(result_root):
    semantic_maps_dict = load_map_data(result_root)
    latest_semantic_map = list(semantic_maps_dict.values())[-1]["data"]

    # plt.imshow(latest_semantic_map, cmap="gray", origin="lower")
    # plt.title("Difference between Constrained and Unconstrained Maps")
    # plt.show()

    gt_semantic_map = get_gt_semantic_map(result_root)

    latest_semantic_map_binary = np.where(latest_semantic_map > 0, 1, 0)
    gt_semantic_map_binary = np.where(gt_semantic_map > 0, 1, 0)

    gt_semantic_map_area = gt_semantic_map_binary.sum()
    latest_semantic_map_area = latest_semantic_map_binary.sum()

    area_ratio = latest_semantic_map_area / gt_semantic_map_area

    print(f"area ratio: {area_ratio:.4f}")

    intersection = np.logical_and(latest_semantic_map_binary, gt_semantic_map_binary).sum()

    union = np.logical_or(latest_semantic_map_binary, gt_semantic_map_binary).sum()

    iou = intersection / union if union != 0 else 0

    print(f"IoU: {iou:.4f}")

    return iou, area_ratio

if __name__ == '__main__':
    text_query_list = []
    iou_list = []
    area_ratio_list = []
    for result_root in result_roots:
        with open(os.path.join(result_root, "exp_config.json"), "r") as f:
            text_query = json.load(f)["text_queries"][0]
        iou, area_ratio = compare_vlm_map(result_root)
        text_query_list.append(text_query)
        iou_list.append(iou)
        area_ratio_list.append(area_ratio)

    # Set up the positions for the bars
    x = np.arange(len(text_query_list))  # the label locations
    width = 0.35  # the width of the bars

    # Create the plot
    fig, ax = plt.subplots()

    # Plotting IoU bars
    iou_bars = ax.bar(x - width / 2, iou_list, width, label='IoU')

    # Plotting Area Ratio bars
    area_ratio_bars = ax.bar(x + width / 2, area_ratio_list, width, label='Area Ratio')

    # Add labels, title, and custom x-axis tick labels
    ax.set_xlabel('Text Queries')
    ax.set_ylabel('Values')
    ax.set_title('IoU and Area Ratio by Text Query')
    ax.set_xticks(x)
    ax.set_xticklabels(text_query_list)
    ax.legend()


    # Adding value labels on top of each bar for better readability (optional)
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    add_labels(iou_bars)
    add_labels(area_ratio_bars)

    # Adjust the layout to make room for the labels
    fig.tight_layout()

    # Save the plot if needed
    plt.savefig("iou_area_ratio_histogram.png", bbox_inches='tight', dpi=300)

    # Display the plot
    plt.show()