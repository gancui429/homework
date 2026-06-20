import os
import subprocess
import argparse

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run COLMAP for multi-view stereo')
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir

    colmap_exe = r"D:\colmap\COLMAP.bat"

    # 特征提取：修正GPU参数前缀 FeatureExtraction
    subprocess.run([
        colmap_exe, 'feature_extractor',
        '--image_path', os.path.join(data_dir, 'images'),
        '--database_path', os.path.join(data_dir, 'database.db'),
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', 'PINHOLE',
        '--FeatureExtraction.use_gpu', '0'
    ], check=True)

    # 特征匹配：修正GPU参数前缀 FeatureMatching
    subprocess.run([
        colmap_exe, 'exhaustive_matcher',
        '--database_path', os.path.join(data_dir, 'database.db'),
        '--FeatureMatching.use_gpu', '0'
    ], check=True)

    os.makedirs(os.path.join(data_dir, 'sparse'), exist_ok=True)

    # 稀疏重建
    subprocess.run([
        colmap_exe, 'mapper',
        '--image_path', os.path.join(data_dir, 'images'),
        '--database_path', os.path.join(data_dir, 'database.db'),
        '--output_path', os.path.join(data_dir, 'sparse')
    ], check=True)

    # 模型转文本
    os.makedirs(os.path.join(data_dir, 'sparse', '0_text'), exist_ok=True)
    subprocess.run([
        colmap_exe, 'model_converter',
        '--input_path', os.path.join(data_dir, 'sparse', '0'),
        '--output_path', os.path.join(data_dir, 'sparse', '0_text'),
        '--output_type', 'TXT'
    ], check=True)

    print("COLMAP 多视图重建流程执行完成！")
    print("稀疏重建结果保存路径：", os.path.join(data_dir, 'sparse', '0_text'))