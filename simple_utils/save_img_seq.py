import os
import re
import cv2
import imageio




def save_regnerf_sequence(save_dir, filename, img_dir, matcher, save_format='gif', fps=1):
        assert save_format in ['gif', 'mp4']
        if not filename.endswith(save_format):
            filename += f".{save_format}"
        matcher = re.compile(matcher)
        img_dir = os.path.join(save_dir, img_dir)
        imgs = []
        for f in os.listdir(img_dir):
            if matcher.search(f):
                imgs.append(f)
        imgs = sorted(imgs, key=lambda f: int(matcher.search(f).groups()[0]))
        imgs = [cv2.imread(os.path.join(img_dir, f)) for f in imgs]
        save_path = os.path.join(save_dir, filename)
        if save_format == 'gif':
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            imageio.mimsave(save_path, imgs, fps=fps, palettesize=256)
        elif save_format == 'mp4':
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            imageio.mimsave(save_path, imgs, fps=fps)
        
            
if __name__ == "__main__":
    save_regnerf_sequence("/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/exp/transient-neus-blender-lego", "regnerf", "/scratch/ondemand28/weihanluo/multiview_transient_project/instant-nsr-pl/exp/transient-neus-blender-lego/lego-two-views@20240226-191546/save", "(\d+)\.png", "mp4")