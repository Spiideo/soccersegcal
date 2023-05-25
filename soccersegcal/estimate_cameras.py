from soccersegcal.dataloader import SoccerNetFieldSegmentationDataset
from soccersegcal.pose import segs2cam
import torch
from sncalib.baseline_cameras import Camera
import numpy as np
from pathlib import Path
import json
from time import time
from scipy.optimize import fmin
from soccersegcal.train import LitSoccerFieldSegmentation
import fire

def main(checkpoint_path="checkpoint.ckpt", indexes=None, data=None, out='cams_out', part='valid', show=False, overwrite=False):
    out_dir = Path(out)

    (out_dir / part).mkdir(parents=True, exist_ok=True)

    world_scale = 100
    if data is None:
        data = SoccerNetFieldSegmentationDataset(width=960, split=part)
    if indexes is None:
        indexes = range(len(data))
    device = torch.device('cuda')

    segmentation_model = LitSoccerFieldSegmentation.load_from_checkpoint(checkpoint_path).to(memory_format=torch.channels_last, device=device)
    segmentation_model.eval()

    for i in indexes:
        print("Index:", i)
        start_time = time()
        entry = data[i]
        ofn = out_dir / part / ('camera_' + entry['name'].replace('.jpg', '.json'))
        pfn = out_dir / part / f"camera_{int(entry['name'].split('.')[0]) - 1:05d}.json"
        if ofn.exists() and not overwrite:
            continue

        if pfn.exists():
            prev_cam = Camera()
            prev_cam.from_json_parameters(json.load(pfn.open()))
            prev_cam.position /= world_scale
            assert prev_cam.image_width == entry['image'].shape[-1]
        else:
            prev_cam = None


        if checkpoint_path is None:
            segs = entry['segments']
        else:
            with torch.no_grad():
                segs = torch.sigmoid_(segmentation_model(entry['image'].to(device)[None]))[0].cpu()
        ptz_model = segs2cam(segs, world_scale, prev_cam, show=show)
        if ptz_model is None:
            continue

        ptz_model = ptz_model.cpu()
        smalles_image_side = min(segs.shape[2], segs.shape[1])
        f = smalles_image_side / 2 / ptz_model.camera_focal.item()
        cam = Camera(segs.shape[2], segs.shape[1])
        cam.from_json_parameters({
            'position_meters': ptz_model.camera_position.detach().numpy() * world_scale,
            'principal_point': cam.principal_point,
            'x_focal_length': f,
            'y_focal_length': f,
            'pan_degrees': np.rad2deg(ptz_model.camera_pan.item()),
            'tilt_degrees': np.rad2deg(ptz_model.camera_tilt.item()),
            'roll_degrees': np.rad2deg(ptz_model.camera_roll.item()),
            'radial_distortion': ptz_model.radial_distortion.detach().numpy() if hasattr(ptz_model, 'radial_distortion') else np.zeros(6),
            'tangential_distortion': ptz_model.tangential_disto.detach().numpy() if hasattr(ptz_model, 'tangential_disto') else np.zeros(2),
            'thin_prism_distortion': ptz_model.thin_prism_disto.detach().numpy() if hasattr(ptz_model, 'thin_prism_disto') else np.zeros(4),
        })
        cam.scale_resolution(960 / data.shape[1])
        with open(ofn, "w") as fd:
            params = cam.to_json_parameters()
            if hasattr(ptz_model, 'mode_coeffs'):
                params['field_length'] = 105 + ptz_model.mode_coeffs[0].item() * 2 * world_scale
                params['field_width'] = 68 + ptz_model.mode_coeffs[1].item() * 2 * world_scale
            json.dump(params, fd)
        print("    ", time() - start_time, "s")



if __name__ == '__main__':
    fire.Fire(main)