if __name__ == '__main__':
    import shutil
    import os
    import random

    to_split_dir = r'E:\my_files\programmes\python\super_resolution_images\new'
    target_dirs = [
        r'E:\my_files\programmes\python\super_resolution_images\fold0',
        r'E:\my_files\programmes\python\super_resolution_images\fold1',
        r'E:\my_files\programmes\python\super_resolution_images\fold2',
        r'E:\my_files\programmes\python\super_resolution_images\fold3',
        r'E:\my_files\programmes\python\super_resolution_images\fold4',
    ]

    all_to_move_imgs = os.listdir(to_split_dir)
    random.shuffle(all_to_move_imgs)
    for idx, file in enumerate(all_to_move_imgs):
        src = os.path.join(to_split_dir, file)
        dst = os.path.join(target_dirs[idx % len(target_dirs)], file)
        print('src:', src)
        print('dst:', dst)
        shutil.move(src, dst)
