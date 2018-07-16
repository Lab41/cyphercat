import os 

train_dir = 'datasets/tiny-imagenet-200/train'
class_dirs = [os.path.join(train_dir, o) for o in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,o))]

for c in class_dirs: 
    for f in os.listdir(c+"/images"):
        os.rename(c+"/images/"+f, c+"/"+f)
    for d in os.listdir(c): 
        if d.find("JPEG") == -1: 
            if os.path.isfile(c+"/"+d): 
                os.remove(c+"/"+d)
            elif os.path.isdir(c+"/"+d): 
                os.rmdir(c+"/"+d)
            

            
with open('datasets/tiny-imagenet-200/val/val_annotations.txt') as f: 
    content = f.readlines()

    

for x in content: 
    line = x.split()
    
    if not os.path.exists('datasets/tiny-imagenet-200/val/'+line[1]): 
        os.makedirs('datasets/tiny-imagenet-200/val/'+line[1])
    
    new_file_name = 'datasets/tiny-imagenet-200/val/'+line[1]+'/'+line[0] 
    old_file_name = 'datasets/tiny-imagenet-200/val/images/'+line[0]
    os.rename(old_file_name, new_file_name)

    