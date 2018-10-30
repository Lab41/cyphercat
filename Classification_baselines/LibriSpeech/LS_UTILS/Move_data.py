import os

def move_files(abs_dirname,speaker,datapath,trainSet,cvSet,tstSet):
    """Move files into subdirectories."""

    for subdir in os.listdir(abs_dirname):
        files = [os.path.join(abs_dirname,subdir, f) for f in os.listdir(os.path.join(abs_dirname,subdir))]  
        cv_dir = os.path.abspath(os.path.join(datapath + cvSet  + speaker))
        cv_subdir = os.path.join(cv_dir,subdir)
        test_dir = os.path.abspath(os.path.join(datapath + tstSet  + speaker))
        test_subdir = os.path.join(test_dir,subdir)
        if not os.path.isdir(test_subdir):
            if not os.path.isdir(test_dir):
                print('splitting',speaker)
                os.mkdir(cv_dir)
                os.mkdir(test_dir)
            os.mkdir(cv_subdir)
            os.mkdir(test_subdir)
            
            #separate files
            # 1 test
            # 2 cv
            # 3,4,5 train
            # 6 test
            # 7 cv
            # 8,9,0 train
            ncv = [2,7]
            ntest = [1,6]
              
            for f in files:
                num = f[-9:-5]
                if num != 'tran':
                    rem = int(num) % 10

                    if rem in ncv: #move to cv

                    # move file to target dir
                        f_base = os.path.basename(f)
                        shutil.move(f, cv_subdir)
                    elif rem in ntest:

                    # move file to target dir
                        f_base = os.path.basename(f)
                        shutil.move(f, test_subdir)

def main(speakers,datapath,trainSet,cvSet,tstSet):

    for speaker in speakers:
        src_dir = datapath + trainSet + speaker

        if not os.path.exists(src_dir):
            raise Exception('Directory does not exist ({0}).'.format(src_dir))

        move_files(os.path.abspath(src_dir),speaker,datapath,trainSet,cvSet,tstSet)
