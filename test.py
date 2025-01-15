import argparse
import os
import subprocess
import numpy as np
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from outputs.pfmprocess import resize_pfm, median_normalize, visual_pfm
from PIL import Image
from dust3r.utils.device import to_numpy
from justpfm import justpfm
import cv2
import shutil
import re
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate dust3r model on ETH3D dataset")
    parser.add_argument('dataset_name', type=str, help="Name of the dataset (e.g., ETH3D)")
    parser.add_argument('data_path', type=str, help="Path to the training data")
    parser.add_argument('gt_path', type=str, help="Path to the groundtruth data")
    parser.add_argument('output_path', type=str, help="Path to the output folder")
    parser.add_argument('--visualization', type=bool, default=False, help="Enable visualization (default: False)")
    return parser.parse_args()

def dust3r_reconstruct(data_folder, output_folder):
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory

    for folder_name in os.listdir(data_folder):
        data_path = os.path.join(data_folder, folder_name)
        output_path = os.path.join(output_folder, folder_name)
        im0_path = os.path.join(data_path, 'im0.png')
        im1_path = os.path.join(data_path, 'im1.png')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 清空文件夹 (删除文件夹及其内容)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        # 重新创建空的文件夹
        os.makedirs(output_path)


        images = load_images([im0_path, im1_path], size=512)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
        #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
        # in each view you have:
        # an integer image identifier: view1['idx'] and view2['idx']
        # the img: view1['img'] and view2['img']
        # the image shape: view1['true_shape'] and view2['true_shape']
        # an instance string output by the dataloader: view1['instance'] and view2['instance']
        # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
        # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
        # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

        # next we'll use the global_aligner to align the predictions
        # depending on your task, you may be fine with the raw output and not need it
        # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
        # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
        # scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        # loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)

        # retrieve useful values from scene:
        imgs = scene.imgs
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()

        # 获取深度图数据
        depths = to_numpy(scene.get_depthmaps())  # 将深度图转换为 numpy 数组


        # # 保存每个深度图
        for i, depth in enumerate(depths):
            # 对深度图进行归一化处理 (假设深度图是一个浮动数组)
            if folder_name == 'terrace_1s':
                print(depth)
            depth = - depth + np.max(depth)
            depth_normalized = (depth / depth.max() * 255).astype(np.uint8)  # 将深度归一化到 [0, 255]
            
            depth_image = Image.fromarray(depth_normalized)  # 创建图片对象
            depth_image.save(os.path.join(output_path, f'depth_map_{i}.png'))  # 保存图片
            justpfm.write_pfm(os.path.join(output_path, f'depth_map_{i}.pfm'), depth)  

def compute_loss(output_folder, groundtruth_folder):
    total_loss = {}
    count = 0
    
    # Iterate over all the folders in the data folder
    for folder_name in os.listdir(output_folder):
        output_path = os.path.join(output_folder, folder_name)
        gt_path = os.path.join(groundtruth_folder, folder_name)
        # Construct paths to the pfm files and groundtruth images
        output_file = os.path.join(output_path, 'depth_map_0.pfm')  # Change this to your actual pfm filename
        gt_file = os.path.join(gt_path, 'disp0GT.pfm')  # Groundtruth PFM file
        mask_file = os.path.join(gt_path, 'mask0nocc.png')
        
        if os.path.exists(output_file) and os.path.exists(gt_file) and os.path.exists(mask_file):
            # 假设output_file, gt_file, mask_file是之前定义的文件路径
            resize_pfm(output_file, gt_file, output_file)
            median_normalize(output_file, gt_file)
            visual_pfm(output_file, os.path.join(output_path, 'visualization.png'))
            visual_pfm(gt_file, os.path.join(gt_path, 'visualization.png'))
            result = subprocess.run(['./datasets/ETH3D/main', output_file, gt_file, mask_file, output_path, 'true'],
                                    capture_output=True, text=True)

            # 检查main程序是否遇到了错误
            if result.returncode != 0:
                # 中止程序，输出main函数输出的错误信息
                print("Error running main program:")
                print(result.stderr)
                sys.exit(1)
            else:
                # main程序正常比较了两张图片并且输出了loss
                # 初始化字典来存储损失数据
                loss_data = {}

                # 使用正则表达式提取输出中的数值数据
                pattern = r"(\w+): ([\d\.]+)"
                matches = re.findall(pattern, result.stdout)
                
                # 标记当前处理的是哪个部分（non-occluded或All）
                current_section = None
                
                # 遍历输出中的每一行
                for line in result.stdout.splitlines():
                    if "Non-occluded:" in line:
                        current_section = "non-occluded"
                    elif "All:" in line:
                        current_section = "all"
                    elif current_section:
                        # 提取键值对并存储到字典中
                        for key, value in re.findall(pattern, line):
                            loss_data[f"{current_section}:{key}"] = float(value)
                
                # 打印提取的损失数据字典
                print(folder_name)
                for key, value in loss_data.items():
                    print(f"{key}: {value}")
                print('\n')

                # if 'non-occluded:avgerr' in loss_data:
                #     total_loss += loss_data['non-occluded:avgerr']
                #     count += 1
                total_loss = {key: total_loss.get(key, 0) + loss_data.get(key, 0) for key in set(total_loss) | set(loss_data)}
                count += 1
        else:
            print("Wrong path!")
            sys.exit(1)
    
    if count > 0:
        average_loss = {key: total_loss.get(key, 0) / count for key in set(total_loss)}
        return average_loss
    else:
        return None

# def main():

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jiangt/anaconda3/envs/dust/lib
    
if __name__ == "__main__":
    args = parse_arguments()

    # Define paths
    data_folder = args.data_path
    groundtruth_folder = args.gt_path
    output_folder = args.output_path

    # Use the model to compute the output
    dust3r_reconstruct(data_folder, output_folder)

    # Compute loss
    avg_loss = compute_loss(output_folder, groundtruth_folder)
    
    if avg_loss is not None:
        for key, value in avg_loss.items():
            print(key, ':', value)
    else:
        print("No valid data found or unable to compute loss.")

