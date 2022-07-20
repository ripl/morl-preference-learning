import shutil

def clear_videos(pathname):
    try:
        shutil.rmtree(pathname)    
    except:
        pass