import os
import json
import pickle
import urllib
import urllib.request
from tqdm import tqdm

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    return data


def load_pkl(file_path):
    assert os.path.exists(file_path)
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
        
    return data
    
def save_pkl(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle)
        
        
def del_dir(dir_path):
    os.removedirs(dir_path)
    
def create_dir(dir_path, cover=False):
    if cover or not os.path.exists(dir_path):
        if cover and os.path.exists(dir_path):
            os.removedirs(dir_path)
        os.makedirs(dir_path)
        
        
def save_video(video_array, video_save_path):
    import cv2
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    output_movie = cv2.VideoWriter(video_save_path, fourcc, 10, (640, 360))
    
    for frame in video_array:
        output_movie.write(frame)
        
    out.release()
    cv2.destroyAllWindows()
    
def download_helper(url, filename):
    'Download file from given url. Modified from `torchvision.dataset.utils`'
    def gen_bar_updater():
        pbar = tqdm(total=None)

        def bar_update(count, block_size, total_size):
            if pbar.total is None and total_size:
                pbar.total = total_size
            progress_bytes = count * block_size
            pbar.update(progress_bytes - pbar.n)

        return bar_update

    try:
        print('Downloading ' + url + ' to ' + filename)
        urllib.request.urlretrieve(
            url, filename,
            reporthook=gen_bar_updater()
        )
        
        return True
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                    ' Downloading ' + url + ' to ' + filename)
            urllib.request.urlretrieve(
                url, filename,
                reporthook=gen_bar_updater()
            )
            
            return True
        else:
            raise e