import os, shutil

lfw_dir = '../datasets/lfw_original/'
people_dir = os.listdir(lfw_dir)


num_per_class = 15

new_dir = '../datasets/lfw/lfw_'+str(num_per_class)

if not os.path.isdir(new_dir):
    os.makedirs(new_dir)


i = 0
for p in people_dir:
    imgs = os.listdir(os.path.join(lfw_dir,p))
    if len(imgs) >= 15:
        shutil.copytree(os.path.join(lfw_dir,p),os.path.join(new_dir,p))
        i += 1

