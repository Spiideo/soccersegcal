from dataloader import SoccerNetFieldSegmentationDataset, pview
from sncalib.soccerpitch import SoccerPitch
from sncalib.baseline_cameras import Camera
from sncalib.camera import pan_tilt_roll_to_orientation, rotation_matrix_to_pan_tilt_roll
import numpy as np
from pytorch3d.structures import Meshes
import torch
from torch import nn
from vi3o import flipp
from pytorch3d.io import save_obj, load_obj
from itertools import chain
from scipy.optimize import fmin
from copy import deepcopy
from collections import defaultdict
from torchvision.transforms.functional import resize
from time import time
from scipy.spatial import Delaunay
from copy import copy

from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, BlendParams, SoftSilhouetteShader,
)

from pytorch3d.transforms.so3 import so3_exp_map, so3_log_map


def sample_line(p1, p2, mesh_res):
    v = p2 - p1
    l = np.linalg.norm(v)
    return p1 + np.linspace(0, l, int(np.ceil(l / mesh_res)), endpoint=False)[:, None] * v / l

def triangulate_border(border, world_scale, mesh_res):
    center = border.mean(0)
    v = border - center
    flat_dim = border.std(0).argmin()
    if flat_dim == 2:
        alpha = np.arctan2(v[:,1], v[:,0])
    elif flat_dim == 1:
        alpha = np.arctan2(v[:,2], v[:,0])
    else:
        alpha = np.arctan2(v[:,2], v[:,1])
    order = np.argsort(alpha)
    border = border[order]

    if False:
        faces = [[0, ((i-1) % len(border)) + 1, i + 1] for i in range(len(border))]
        vertices = np.vstack([center, border]) / world_scale
        return torch.tensor(faces).to(int), torch.tensor(vertices).to(torch.float32)


    subsampled_borders = []
    for i in range(len(border)):
        subsampled_borders.append(sample_line(border[i - 1], border[i], mesh_res))
    border = np.vstack(subsampled_borders)
    verts = border.copy()
    for pkt in border:
        for p in sample_line(center, pkt, mesh_res):
            if np.sqrt(((verts - p)**2).sum(1).min()) > mesh_res:
                verts = np.vstack([verts, p])

    dims = tuple(i for i in range(3) if i != flat_dim)
    tri = Delaunay(verts[:, dims])
    faces = tri.simplices
    return torch.tensor(faces).to(int), torch.tensor(verts / world_scale).to(torch.float32)

def create_pitch_meshes(pitch, world_scale, mesh_res=5):
    sampled = pitch.sample_field_points(1000, mesh_res/2)
    faces = [torch.zeros((0,3)).to(int)] * 6
    vertices = [torch.zeros((0,3))] * 6
    right_centers = [None] * 6
    for n, area in pitch.field_areas.items():
        if 'Circle' in n:
            border = np.array(sampled[n])
        else:
            point_names = list(set(sum([pitch.line_extremities_keys[b] for b in area['border'] if b in area['contains']], ())))
            border = np.array([pitch.point_dict[n] for n in point_names])
        f, v = triangulate_border(border, world_scale, mesh_res)
        faces[area['index']] = torch.vstack([faces[area['index']], f + len(vertices[area['index']])])
        vertices[area['index']] = torch.vstack([vertices[area['index']], v])
        if 'left' not in n:
            right_centers[area['index']] = border.mean(0)
    save_obj("/tmp/t.obj", vertices[1], faces[1])
    meshes = Meshes(verts=vertices, faces=faces)

    verts = meshes.verts_padded()
    length_mode = torch.zeros_like(verts)
    length_mode[:,:,0] = verts[:,:,0].sign()
    length_mode[1] = 0  # Dont update center cricle
    tmp_mesh = meshes.update_padded(length_mode)
    length_mode = tmp_mesh.verts_packed()

    width_mode = torch.zeros_like(verts)
    width_mode[0,:,1] = verts[0,:,1].sign()
    tmp_mesh = meshes.update_padded(width_mode)
    width_mode = tmp_mesh.verts_packed()
    modes = torch.stack([length_mode, width_mode])

    return meshes, right_centers, modes

def uni(a, b):
    return torch.rand(1) * (b - a) + a

class Model(nn.Module):
    def __init__(self, meshes, silhouette, start_cam):
        super().__init__()
        self.meshes = meshes
        self.register_buffer('image_ref', silhouette)
        self.renderer = self.create_renderer()

        rot = so3_log_map(torch.tensor(start_cam.rotation.T).to('cuda')[None]).to(torch.float32)
        smalles_image_side = min(self.renderer.rasterizer.raster_settings.image_size)
        f = (start_cam.xfocal_length + start_cam.yfocal_length) / smalles_image_side

        self.camera_rotation = nn.Parameter(rot)
        self.camera_position = nn.Parameter(torch.tensor(start_cam.position).to(torch.float32))
        self.camera_focal = nn.Parameter(torch.tensor([1/f]).to(torch.float32))


    def create_renderer(self):
        cameras = FoVPerspectiveCameras(device='cuda')
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=self.image_ref.shape[1:],
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100,
        )
        return MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
            # shader=SoftPhongShader(device='cuda', cameras=cameras, lights=PointLights(device='cuda', location=[[0.0, 0.0, -3.0]]))
        )


    def forward(self):
        if self.meshes.device != self.camera_position.device:
            self.meshes = self.meshes.to(self.camera_position.device)

        R = so3_exp_map(self.camera_rotation)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)

        f = 1/self.camera_focal
        K = torch.diag(torch.hstack([f.repeat(2), torch.tensor([1, 0]).to(f.device)]))
        K[3,2] = -1
        K[2, 3] = 1

        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T, K=K[None])

        # Calculate the silhouette loss
        loss = nn.functional.mse_loss(self.image_ref, image[:, :, :, 3])
        return loss, image[:, :, :, 3]

def torch_pan_tilt_roll_to_orientation(pan, tilt, roll):
    zero = torch.tensor(0).to(pan.device).to(pan.dtype)
    one = zero + 1
    Rpan = torch.vstack([
        torch.hstack([torch.cos(pan), -torch.sin(pan), zero]),
        torch.hstack([torch.sin(pan), torch.cos(pan), zero]),
        torch.hstack([zero, zero, one])])
    Rtilt = torch.vstack([
        torch.hstack([one, zero, zero]),
        torch.hstack([zero, torch.cos(tilt), -torch.sin(tilt)]),
        torch.hstack([zero, torch.sin(tilt), torch.cos(tilt)])])
    Rroll = torch.vstack([
        torch.hstack([torch.cos(roll), -torch.sin(roll), zero]),
        torch.hstack([torch.sin(roll), torch.cos(roll), zero]),
        torch.hstack([zero, zero, one])])
    return torch.bmm(Rpan[None], torch.bmm(Rtilt[None], Rroll[None]))[0]

def torch_pan_tilt_to_orientation(pan, tilt):
    zero = torch.tensor(0).to(pan.device).to(pan.dtype)
    one = zero + 1
    Rpan = torch.vstack([
        torch.hstack([torch.cos(pan), -torch.sin(pan), zero]),
        torch.hstack([torch.sin(pan), torch.cos(pan), zero]),
        torch.hstack([zero, zero, one])])
    Rtilt = torch.vstack([
        torch.hstack([one, zero, zero]),
        torch.hstack([zero, torch.cos(tilt), -torch.sin(tilt)]),
        torch.hstack([zero, torch.sin(tilt), torch.cos(tilt)])])
    return torch.bmm(Rpan[None], Rtilt[None])[0]

def torch_pan_tilt_to_orientations(pan, tilt):
    zero = torch.zeros(len(pan)).to(pan.device)
    one = zero + 1
    Rpan = torch.stack([
        torch.stack([torch.cos(pan), -torch.sin(pan), zero]),
        torch.stack([torch.sin(pan), torch.cos(pan), zero]),
        torch.stack([zero, zero, one])]).permute(2, 0, 1)
    Rtilt = torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, torch.cos(tilt), -torch.sin(tilt)]),
        torch.stack([zero, torch.sin(tilt), torch.cos(tilt)])]).permute(2, 0, 1)
    Rpan.transpose(1,2)
    return torch.bmm(Rpan, Rtilt)


class PanTiltCameras(nn.Module):
    def __init__(self, n, image_shape) -> None:
        super().__init__()
        self.pan = nn.Parameter(torch.zeros(n))
        self.tilt = nn.Parameter(torch.zeros(n))
        self.position = nn.Parameter(torch.zeros(n, 3))
        self.focal = nn.Parameter(torch.ones(n))
        self.cam = FoVPerspectiveCameras(T=torch.zeros(n, 3)).to('cuda')
        assert len(image_shape) == 2
        self.image_shape = image_shape

    def forward(self, points):
        R = torch_pan_tilt_to_orientations(self.pan, self.tilt)
        T = -torch.bmm(R.transpose(1, 2), self.position[:, :, None])[:, :, 0]
        f = 1 / self.focal
        zero = torch.zeros(len(f))[:,None].to(f.device)
        K = torch.diag_embed(torch.hstack([f[:,None].repeat(1,2), zero+1, zero]))
        K[:, 3, 2] = -1
        K[:, 2, 3] = 1
        return self.cam.transform_points_screen(points, image_size=self.image_shape, R=R, T=T, K=K)


class ZScaledMeshRasterizer(MeshRasterizer):
    def transform(self, meshes_world, **kwargs):
        meshes_ndc = super().transform(meshes_world, **kwargs)
        verts = meshes_ndc.verts_padded()
        verts[..., 2] *= 100
        return meshes_ndc.update_padded(new_verts_padded=verts)


class ModelPanTiltRoll(nn.Module):
    def __init__(self, meshes: Meshes, silhouette, start_cam, do_roll=False):
        super().__init__()
        self.meshes = meshes
        self.register_buffer('image_ref', silhouette, persistent=False)
        weights = ((silhouette > 0.5).sum([1,2]) > 0).to(torch.float32)
        self.register_buffer('weights', weights)
        self.renderer = self.create_renderer()
        self.do_roll = do_roll

        if isinstance(start_cam, Camera):
            pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(start_cam.rotation)
            smalles_image_side = min(self.renderer.rasterizer.raster_settings.image_size)
            f = (start_cam.xfocal_length + start_cam.yfocal_length) / smalles_image_side
            self.camera_pan = nn.Parameter(torch.tensor(pan).to(torch.float32))
            self.camera_tilt = nn.Parameter(torch.tensor(tilt).to(torch.float32))
            self.camera_roll = nn.Parameter(torch.tensor(roll).to(torch.float32))
            self.camera_position = nn.Parameter(torch.tensor(start_cam.position).to(torch.float32))
            self.camera_focal = nn.Parameter(torch.tensor([1/f]).to(torch.float32))
        else:
            self.camera_pan = nn.Parameter(torch.tensor(0).to(torch.float32))
            self.camera_tilt = nn.Parameter(torch.tensor(0).to(torch.float32))
            self.camera_roll = nn.Parameter(torch.tensor(0).to(torch.float32))
            self.camera_position = nn.Parameter(torch.tensor([0,0,0]).to(torch.float32))
            self.camera_focal = nn.Parameter(torch.tensor([0]).to(torch.float32))
            if start_cam is not None:
                self.load_state_dict(start_cam, strict=False)


    def create_renderer(self):
        cameras = self.camera()
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=self.image_ref.shape[1:],
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=20,
        )
        return MeshRenderer(
            rasterizer=self.rasterizer(cameras, raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

    def camera(self):
        return FoVPerspectiveCameras(device='cuda')

    def rasterizer(self, cameras, raster_settings):
        return ZScaledMeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings,
            )

    def render_args(self):
        if self.do_roll:
            R = torch_pan_tilt_roll_to_orientation(self.camera_pan, self.camera_tilt, self.camera_roll)[None]
        else:
            R = torch_pan_tilt_to_orientation(self.camera_pan, self.camera_tilt)[None]
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)

        f = 1/self.camera_focal
        K = torch.diag(torch.hstack([f.repeat(2), torch.tensor([1, 0]).to(f.device)]))
        K[3,2] = -1
        K[2, 3] = 1

        return dict(R=R, T=T, K=K[None])


    def forward(self):
        if self.meshes.device != self.camera_position.device:
            self.meshes = self.meshes.to(self.camera_position.device)
        return self.forward_meshes(self.meshes)

    def forward_meshes(self, meshes):
        image = self.renderer(meshes_world=meshes.clone(), **self.render_args())
        losses = [nn.functional.mse_loss(self.image_ref[i], image[i, :, :, 3]) for i in range(len(image))]
        weights = self.weights
        loss = torch.sum(weights * torch.hstack(losses)) / weights.sum()
        return loss, image[:, :, :, 3]

class ModelPanTiltRollModes(ModelPanTiltRoll):
    def __init__(self, meshes, modes, silhouette, start_cam, do_roll=False):
        if isinstance(start_cam, Camera):
            super().__init__(meshes, silhouette, start_cam, do_roll)
            self.mode_coeffs = nn.Parameter(torch.zeros(len(modes)))
        else:
            super().__init__(meshes, silhouette, None, do_roll)
            self.mode_coeffs = nn.Parameter(torch.zeros(len(modes)))
            self.load_state_dict(start_cam, strict=False)
        self.register_buffer("modes", modes)

    def forward(self):
        if self.meshes.device != self.camera_position.device:
            self.meshes = self.meshes.to(self.camera_position.device)
        offsets = (self.mode_coeffs[:, None, None] * self.modes).sum(0)
        meshes = self.meshes.offset_verts(offsets)
        return self.forward_meshes(meshes)


class DualTransform:
    def __init__(self, pre_tran, post_tran) -> None:
        self.pre_tran = pre_tran
        self.post_tran = post_tran

    def transform_points(self, pkt, eps):
        pkt = self.pre_tran.transform_points(pkt, eps=eps)
        pkt = self.post_tran.transform_points(pkt, eps=eps)
        return pkt


class DistortedTransform:
    def __init__(self, pre_tran, camera_kwargs) -> None:
        self.pre_tran = pre_tran
        self.radial_distortion = camera_kwargs['radial_distortion']
        self.tangential_disto = camera_kwargs['tangential_disto']
        self.thin_prism_disto = camera_kwargs['thin_prism_disto']

    def transform_points(self, verts_world, eps):
        verts = self.pre_tran.transform_points(verts_world, eps=eps)

        numerator = 1
        denominator = 1
        norm_verts = verts / verts[:, :, 2:3]
        radius = torch.sqrt(norm_verts[:, :, 0] * norm_verts[:, :, 0] + norm_verts[:, :, 1] * norm_verts[:, :, 1])

        for i in range(3):
            k = self.radial_distortion[i]
            numerator += k * radius ** (2 * (i + 1))
            k2n = self.radial_distortion[i + 3]
            denominator += k2n * radius ** (2 * (i + 1))

        radial_distortion_factor = numerator / denominator

        xpp = norm_verts[:, :, 0] * radial_distortion_factor + \
              2 * self.tangential_disto[0] * norm_verts[:, :, 0] * norm_verts[:, :, 1] + self.tangential_disto[1] * (radius ** 2 + 2 * norm_verts[:, :, 0] ** 2)  + \
              self.thin_prism_disto[0] * radius ** 2 + self.thin_prism_disto[1] * radius ** 4
        ypp = norm_verts[:, :, 1] * radial_distortion_factor + \
              2 * self.tangential_disto[1] * norm_verts[:, :, 0] * norm_verts[:, :, 1] + self.tangential_disto[0] * (radius ** 2 + 2 * norm_verts[:, :, 1] ** 2)  + \
              self.thin_prism_disto[2] * radius ** 2 + self.thin_prism_disto[3] * radius ** 4
        new_verts = torch.stack([xpp * verts[:, :, 2], ypp * verts[:, :, 2], verts[:, :, 2]], 2)
        verts = torch.where(radius[:, :, None] < 10, new_verts, verts)  # Points far outside the image will wrap around the image if distorted, let's just ignore them

        return verts

    def compose(self, other):
        return DualTransform(self, other)


class DistortedFoVPerspectiveCameras(FoVPerspectiveCameras):
    def get_world_to_view_transform(self, **kwargs):
        rot_tran = super().get_world_to_view_transform(**kwargs)
        return DistortedTransform(rot_tran, kwargs)

    def transform_points(self, verts_world, eps):
        return None  # We can't tranforms without distortion parameters

class ModelPanTiltRollDistorted(ModelPanTiltRoll):
    def __init__(self, meshes, silhouette, start_cam):
        if isinstance(start_cam, Camera):
            super().__init__(meshes, silhouette, start_cam, True)
            self.radial_distortion = nn.Parameter(torch.tensor(start_cam.radial_distortion).to(torch.float32))
            self.tangential_disto = nn.Parameter(torch.tensor(start_cam.tangential_disto).to(torch.float32))
            self.thin_prism_disto = nn.Parameter(torch.tensor(start_cam.thin_prism_disto).to(torch.float32))
        else:
            super().__init__(meshes, silhouette, None, True)
            self.radial_distortion = nn.Parameter(torch.tensor([0] * 6).to(torch.float32))
            self.tangential_disto = nn.Parameter(torch.tensor([0] * 2).to(torch.float32))
            self.thin_prism_disto = nn.Parameter(torch.tensor([0] * 4).to(torch.float32))
            self.load_state_dict(start_cam, strict=False)

    def camera(self):
        return DistortedFoVPerspectiveCameras(device='cuda')

    def render_args(self):
        kwargs = super().render_args()
        kwargs['radial_distortion'] = self.radial_distortion
        kwargs['tangential_disto'] = self.tangential_disto
        kwargs['thin_prism_disto'] = self.thin_prism_disto
        return kwargs


def move_camera(meshes, segs, cam, world_scale, direction='pan'):
    model = Model(meshes, segs, cam).to('cuda')
    with torch.no_grad():
        while True:
            for t in chain(range(100), range(100, 0, -1)):
                if direction == 'pan':
                    t *= np.pi / 180
                    rotation = np.transpose(pan_tilt_roll_to_orientation(t, np.pi/4, 0))
                    model.camera_rotation[:] = so3_log_map(torch.tensor(rotation.T).to('cuda')[None]).to(torch.float32)
                elif direction == 'tilt':
                    t *= np.pi / 180
                    rotation = np.transpose(pan_tilt_roll_to_orientation(0, t, 0))
                    model.camera_rotation[:] = so3_log_map(torch.tensor(rotation.T).to('cuda')[None]).to(torch.float32)
                elif direction == 'z':
                    t /= world_scale
                    print(t)
                    model.camera_position[:] = torch.tensor([0, 0, -t]).cuda()
                else:
                    raise NotImplementedError
                loss, image = model()
                image_view = image[:3] + image[3:]
                pview(image_view, pause=True)

def optimize_camera(meshes, segs, cam, max_itter=10000, max_no_improve=10000, min_loss=0.001, roll=False, distort=False, show=True, modes=None, seg_indexes=None, lr=0.001):
    if modes is None:
        if not distort:
            model = ModelPanTiltRoll(meshes, segs, cam, do_roll=roll).to('cuda')
        else:
            assert roll
            model = ModelPanTiltRollDistorted(meshes, segs, cam).to('cuda')
            # model.tangential_disto.requires_grad = False
            # model.thin_prism_disto.requires_grad = False
            # model.radial_distortion.requires_grad = False
    else:
        assert not distort
        model = ModelPanTiltRollModes(meshes, modes, segs, cam, do_roll=roll).to('cuda')

    if seg_indexes is not None:
        model.weights[:] = 0.0
        model.weights[seg_indexes] = 1.0
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.1)

    best_loss = float('inf')
    last_improve = 0
    converged = False
    # print(model.camera_focal.item(), model.camera_pan.item(), model.camera_tilt.item(), model.camera_position.detach().cpu().numpy())
    for i in range(max_itter):
        optimizer.zero_grad()
        loss, image = model()
        loss.backward()
        optimizer.step()

        if loss < min_loss:
            break
        if loss < best_loss:
            last_improve = i
            best_loss = loss.item()
        if i - last_improve > max_no_improve:
            converged = True
            break

        if show:
            image_view = image[:3] + image[3:]
            segs_view = segs[:3] + segs[3:]
            pview(image_view, pause=False)
            pview(segs_view, pause=False)
            pview(segs_view.to('cuda')-image_view, pause=False)
            # from vi3o.debugview import DebugViewer
            # out.view(imscale(np.hstack([a[0] for a in DebugViewer.named_viewers['Default'].image_array]), (720, 134)))

            flipp() #pause=True)

    return loss, model, converged

# from vi3o.image import ImageDirOut, imscale
# out = ImageDirOut("demo", "png")

def segs2cam(segs, world_scale, additional_start_cam=None, *, show=False):
    do_roll = True
    pitch = SoccerPitch()
    meshes, right_centers, modes = create_pitch_meshes(pitch, world_scale)
    right_centers = torch.tensor(np.array(right_centers, np.float32)).to('cuda') / world_scale

    start_pos, pans = load_start_positions()
    nominal_shape = [144, 256]
    cameras = PanTiltCameras(len(start_pos), nominal_shape).to('cuda')
    with torch.no_grad():
        cameras.position[:] = torch.tensor(start_pos) / world_scale
        cameras.tilt[:] = np.pi / 4
        cameras.pan[:] = torch.tensor(pans)
        cameras.focal[:] = 2 * 20 / min(cameras.image_shape)
        cameras.position.requires_grad = False

    sx, sy = np.array(nominal_shape) / segs.shape[-2:]
    assert np.abs(sx - sy) < 1e-6

    seg_centers = []
    seg_sizes = []
    for s in segs:
        indexes = torch.nonzero(s > 0.5).float()
        seg_sizes.append(len(indexes))
        if len(indexes) > 0:
            seg_centers.append(sx * indexes.median(0).values.numpy()[::-1])
        else:
            seg_centers.append(None)

    if all(s == 0 for s in seg_sizes[1:]):
        indexes = start_indexes = [0]
    else:
        indexes = [i for i, s in enumerate(seg_sizes) if s > 0]
        start_indexes = [i for i in indexes if i > 0]

    if len(indexes) == 1:
        cameras.focal.requires_grad = False

    right_centers = right_centers[start_indexes]
    seg_centers = torch.tensor(np.array([c for i, c in enumerate(seg_centers) if i in start_indexes])).cuda()

    optimizer = torch.optim.AdamW(cameras.parameters(), lr=0.01)
    for stp in range(500):
        optimizer.zero_grad()
        pkt = cameras(right_centers)
        losses = ((pkt[:, :, :2] - seg_centers[None])**2).sum(2).sum(1)
        losses.sum().backward()
        optimizer.step()

    scale_down = 1
    while segs.shape[-1] // scale_down > 120:
        scale_down *= 2
    segs_lowres = resize(segs, tuple(np.array(segs.shape[-2:]) // scale_down), antialias=True)

    start_cams = []
    if additional_start_cam is not None:
        additional_start_cam = copy(additional_start_cam)
        additional_start_cam.scale_resolution(1/scale_down)
        start_cams.append((-1, additional_start_cam))

    for idx in losses.sort().indices[:5]:
        state_dict = {"camera_" + k: v[idx] for k, v in cameras.state_dict().items()}
        state_dict["camera_focal"] =  state_dict["camera_focal"][None]
        state_dict["camera_roll"] = torch.tensor(0.0)
        start_cams.append((losses[idx], state_dict))

    best_loss = torch.tensor(float('Inf'))
    for start_loss, state_dict in start_cams:
        if start_loss > 2000 and torch.isfinite(best_loss):
            break
        loss, model, converged = optimize_camera(meshes, segs_lowres, state_dict, max_itter=500, show=show, roll=do_roll, seg_indexes=indexes)
        loss, model, converged = optimize_camera(meshes, segs_lowres, model.state_dict(), max_itter=1500, show=show, roll=do_roll, seg_indexes=indexes)
        if loss < best_loss and start_loss > 0:
            best_loss = loss
            best_model = model
            if best_loss < 0.005:
                break

    loss = best_loss
    max_itter = 500
    distort = False
    my_modes = None
    lr = 0.001
    while scale_down >= 1:
        my_meshes = meshes
        segs_lowres = resize(segs, tuple(np.array(segs.shape[-2:]) // scale_down), antialias=True)
        prev_loss = float('Inf')
        while loss / prev_loss < 0.9:
            prev_loss = loss
            loss, best_model, converged = optimize_camera(my_meshes, segs_lowres, best_model.state_dict(), max_itter=max_itter, show=show, roll=do_roll, distort=distort, modes=my_modes, lr=lr)
        max_itter = max(max_itter//2, 50)
        scale_down //= 2
        # distort = True
        # my_modes = modes
        model.weights[:] = 1.0
        # lr /= 2
    return best_model

def load_start_positions():
    long_start = np.load("stats/long_side_start_positions.npy")
    long_pan = [0] * len(long_start) + [np.pi] * len(long_start)
    long_start = np.vstack([long_start, long_start * [1, -1, 1]])
    short_start = np.load("stats/short_side_start_positions.npy")
    short_pan = [3*np.pi/2] * len(short_start)
    start = np.vstack([short_start, long_start])
    pan = np.hstack([short_pan, long_pan])
    return start, pan

def show_camera_view():
    world_scale = 100
    pitch = SoccerPitch()
    data = SoccerNetFieldSegmentationDataset(width=256)
    i = 1287
    entry = data[i]
    segs = entry['segments']
    meshes, _, _ = create_pitch_meshes(pitch, world_scale)

    cameras = PanTiltCameras(1, segs.shape[-2:]).to('cuda')
    for pos, pan in zip(*load_start_positions()):
        with torch.no_grad():
            cameras.position[:] = torch.tensor([pos]) / world_scale
            cameras.tilt[:] = np.pi / 4
            cameras.pan[:] = pan
            cameras.focal[:] = 2 * 100 / min(cameras.image_shape)
        idx = 0
        state_dict = {"camera_" + k: v[idx] for k, v in cameras.state_dict().items()}
        state_dict["camera_focal"] = state_dict["camera_focal"][None]
        self = ModelPanTiltRoll(meshes, segs, state_dict).to('cuda')
        loss, image = self()
        image_view = image[:3, :, :] + image[3:, :, :]
        pview(image_view, pause=True); flipp()


def count_x_in_order(centers, order):
    pos = [centers[i][0] for i in order if centers[i] is not None]
    cnt = 0
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            if pos[j] > pos[i]:
                cnt += 1
    return cnt

def guess_initial_camera(pitch, segs):
    seg_centers = []
    for s in segs:
        indexes = torch.nonzero(s > 0).float()
        if len(indexes) > 0:
            seg_center = indexes.mean(0).numpy()[::-1]
        else:
            seg_center = None
        seg_centers.append(seg_center)

    seg_order = [0, 3, 2, 4, 5]

    overlap = segs[0][segs[5]==1]
    if torch.sum(overlap==1) <= torch.sum(segs[5]==1)/2:
        start_position = [0, 35, -20]
        start_pan = 0
        start_tilt = 0
        if count_x_in_order(seg_centers, seg_order) > count_x_in_order(seg_centers, seg_order[::-1]):
            try_first = 'right'
        else:
            try_first = 'left'
    else:
        start_position = [-60, 0, -20]
        start_pan = np.pi / 2
        start_tilt = np.pi/4
        try_first = 'left'

    cam = Camera(segs.shape[2], segs.shape[1])
    cam.from_json_parameters({
        'position_meters': start_position,
        'principal_point': cam.principal_point,
        'x_focal_length': 5e2,
        'y_focal_length': 5e2,
        'pan_degrees': 0,
        'tilt_degrees': 0,
        'roll_degrees': 0,
        'radial_distortion': cam.radial_distortion,
        'tangential_distortion': cam.tangential_disto,
        'thin_prism_distortion': cam.thin_prism_disto,

    })
    initial_cams = {}

    samples = pitch.sample_field_points(1e6, np.pi/8)
    for n, area in pitch.field_areas.items():
        keys = set()
        circles = []
        for l in area['contains']:
            if l in pitch.line_extremities_keys:
                for k in pitch.line_extremities_keys[l]:
                    keys.add(k)
            else:
                circles.extend(samples[l])
        world_center = np.mean([pitch.point_dict[k] for k in keys] + circles, 0)
        seg_center = seg_centers[area['index']]
        if seg_center is not None and n != 'Full field':
            cam = deepcopy(cam)
            def loss(args):
                pan, tilt = args
                cam.rotation = np.transpose(pan_tilt_roll_to_orientation(pan, tilt, 0))
                return np.linalg.norm(cam.project_point(world_center)[:2] - seg_center)
            pan_tilt = fmin(loss, [start_pan, start_tilt], disp=0)
            l = loss(pan_tilt)  # This will update cam with the optimal pan/tilt values
            initial_cams[n] = cam
            # print(n, area['index'], world_center, seg_center, pan_tilt, l)

    cam_order = [n for n in initial_cams.keys() if try_first in n] + [n for n in initial_cams.keys() if try_first not in n]
    return [initial_cams[n] for n in cam_order]


def find_bad_init():
    world_scale = 50
    th = 0.003

    pitch = SoccerPitch()
    data = SoccerNetFieldSegmentationDataset(width=256)
    print(len(data))
    states = defaultdict(dict)

    for index in range(len(data)):
        # index = 8
        entry = data[index]
        segs = entry['segments']

        lines = data.lines(index)
        extremities = {n: [v[0], v[-1]] for n, v in lines.items()}


        initial_cams = guess_initial_camera(pitch, segs)
        states[index]['initial_cams'] = initial_cams
        if len(initial_cams) == 0:
            states[index]['state'] = 'Bad init'
        else:
            meshes, _ = create_pitch_meshes(pitch, world_scale)
            cam = initial_cams[0]
            cam.position /= world_scale
            loss, model, converged = optimize_camera(meshes, segs, cam, max_no_improve=100, min_loss=th, max_itter=3000, show=True)
            states[index]['start_cam_index'] = 0
            states[index]['loss'] = loss.item()
            states[index]['model_state'] = model.state_dict()
            states[index]['cam'] = cam
            if loss <= th:
                states[index]['state'] = 'OK'
            elif converged:
                states[index]['state'] = 'Failed'
            else:
                states[index]['state'] = 'In progress'
            torch.save(states, "states.pth")
            print(index, loss.item(), states[index]['state'])

def find_bad_cont():
    world_scale = 50
    th = 0.003

    pitch = SoccerPitch()
    data = SoccerNetFieldSegmentationDataset(width=256)
    meshes, _ = create_pitch_meshes(pitch, world_scale)
    states = torch.load("states.pth")

    for index, state in sorted(states.items()):
        entry = data[index]
        segs = entry['segments']
        print(index, state['loss'], state['state'])
        if state['state'] == 'OK':
            continue
        elif state['state'] == 'Bad':
            continue
        elif state['state'] == 'In progress':
            loss, model, converged = optimize_camera(meshes, segs, state['model_state'], max_no_improve=100, min_loss=th, max_itter=3000, show=False)
        elif state['state'] == 'Failed':
            state['start_cam_index'] += 1
            if state['start_cam_index'] >= len(state['initial_cams']):
                state['state'] = 'Bad'
                torch.save(states, "states_cont.pth")
                print('    ', loss.item(), state['state'])
                continue
            cam = state['initial_cams'][state['start_cam_index']]
            cam.position /= world_scale
            loss, model, converged = optimize_camera(meshes, segs, cam, max_no_improve=100, min_loss=th, max_itter=3000, show=False)
        else:
            raise NotImplementedError

        state['loss'] = loss.item()
        state['model_state'] = model.state_dict()
        state['cam'] = cam
        if loss <= th:
            state['state'] = 'OK'
        elif converged:
            state['state'] = 'Failed'
        else:
            state['state'] = 'In progress'
        torch.save(states, "states_cont.pth")
        print('    ', loss.item(), state['state'])



def get_pan_tilt_from_direction(direction):
    x, y, z = direction
    return np.arctan2(-y, x), np.arctan2(z, np.sqrt(x ** 2 + y ** 2))


def compare_overlap_check():
    data = SoccerNetFieldSegmentationDataset(width=256)
    states = torch.load("states.pth")

    cnt = defaultdict(int)
    for index in data.bad:
        s = states[index]['state']
        cnt[s] += 1
        if s == 'OK':
            print(index, 'OK')
    print(cnt)


if __name__ == '__main__':
    # find_bad_init()
    # find_bad_cont()
    # compare_overlap_check()
    show_camera_view()


