#############################################################################
# Camera utils
#############################################################################

import torch
from torch.nn.functional import normalize


class CameraExtrinsics:
    def __init__(self, eye, at, up):
        self.num_cameras = len(eye)
        if eye.ndim == 1:
            eye = eye.unsqeueeze(0)
        if at.ndim == 1:
            at = at.unsqueeze(0)
        if up.ndim == 1:
            up = up.unsqueeze(0)
        backward = at - eye
        backward = torch.nn.functional.normalize(backward, dim=-1)
        right = torch.cross(backward, up, dim=-1)
        right = torch.nn.functional.normalize(right, dim=-1)
        up = torch.cross(right, backward, dim=-1)
        self.R = torch.stack((right, up, -backward), dim=1)
        self.t = -self.R @ eye.unsqueeze(-1)

    def transform(self, points):
        num_points = points.shape[-2]
        points = points.expand(self.num_cameras, num_points, 3)[..., None]
        R = self.R[:, None].expand(self.num_cameras, num_points, 3, 3)
        t = self.t[:, None].expand(self.num_cameras, num_points, 3, 1)
        return (R @ points + t).squeeze(-1)

    def inv_transform_rays(self, ray_orig, ray_dir):
        num_rays = ray_dir.shape[-2]
        d = ray_dir.expand(self.num_cameras, num_rays, 3)[..., None]
        o = ray_orig.expand(self.num_cameras, num_rays, 3)[..., None]
        R = self.R[:, None].expand(self.num_cameras, num_rays, 3, 3)
        R_T = R.transpose(2, 3)
        t = self.t[:, None].expand(self.num_cameras, num_rays, 3, 1)
        transformed_dir = R_T @ d
        transformed_orig = R_T @ (o - t)
        return transformed_orig.squeeze(-1), transformed_dir.squeeze(-1)

class CameraIntrinsics:
    def __init__(self, fov, height, width, x0, y0):
        tanHalfAngle = torch.tan(fov / 2.0)
        aspect = height / 2.0
        self.height = height
        self.width = width
        self.focal_x = width / (2 * tanHalfAngle)
        self.focal_y = height / (2 * tanHalfAngle)
        self.near = 1e-2
        self.far = 1e2
        self.x0 = x0
        self.y0 = y0
        self.num_cameras = len(fov)

        self.device = fov.device
        self.dtype = fov.dtype

    def perspective_matrix(self,):
        zero = torch.zeros_like(self.focal_x)
        one = torch.ones_like(self.focal_x)
        rows = [
            torch.stack([self.focal_x, zero,           -self.x0,    zero],       dim=-1),
            torch.stack([zero,         self.focal_y,   -self.y0,    zero],       dim=-1),
            torch.stack([zero,         zero,            zero,       one],        dim=-1),
            torch.stack([zero,         zero,            one,        zero],       dim=-1)
        ]
        persp_mat = torch.stack(rows, dim=1)
        return persp_mat

    def ndc_matrix(self,):
        top = self.height / 2
        bottom = -top
        right = self.width / 2
        left = -right

        tx = -(right + left) / (right - left)
        ty = -(top + bottom) / (top - bottom)

        U = -2.0 * self.near * self.far / (self.far - self.near)
        V = -(self.far + self.near) / (self.far - self.near)
        ndc_mat = torch.tensor([
            [2.0 / (right - left),  0.0,                   0.0,            -tx ],
            [0.0,                   2.0 / (top - bottom),  0.0,            -ty ],
            [0.0,                   0.0,                   U,               V  ],
            [0.0,                   0.0,                   0.0,            -1.0]
        ], dtype=self.dtype, device=self.device)
        return ndc_mat.unsqueeze(0)

    def projection_matrix(self,):
        perspective_matrix = self.perspective_matrix()
        ndc = self.ndc_matrix()
        return ndc @ perspective_matrix

    @staticmethod
    def get_homogeneous_coordinates(points):
        if points.shape[-1] == 4:
            return points
        return torch.cat([points, torch.ones_like(points[..., 0:1])], dim=-1)

    def project(self, points):
        # points shape cab be ([num_cameras], num_points, 4), ([num_cameras], num_points, 3)
        num_points = points.shape[-2]
        homogeneous_points = self.get_homogeneous_coordinates(points)
        homogeneous_points = homogeneous_points.expand(self.num_cameras, num_points, 4)[..., None]
        
        proj = self.projection_matrix()
        proj = proj[:, None].expand(self.num_cameras, num_points, 4, 4)
        return (proj @ homogeneous_points).squeeze(-1)

class Camera:
    def __init__(self, eye, at, up, fov, width, height,):
        self.fov = fov
        self.tan_half_fov = torch.tan(self.fov / 2.0)
        self.height = height
        self.width = width
        self.x0 = torch.zeros_like(eye[..., 0])
        self.y0 = torch.zeros_like(eye[..., 0])
        self.intrinsics = CameraIntrinsics(
            self.fov,
            self.height,
            self.width,
            self.x0,
            self.y0,
        )
        self.extrinsics = CameraExtrinsics(eye, at, up)

    def __len__(self):
        return len(self.x0)

#############################################################################
# Render utils
#############################################################################

import torch
import nvdiffrast
import nvdiffrast.torch as dr
from torch.nn.functional import normalize

def interpolate_attributes(rast_out, rast_out_db, mesh):
    normals = nvdiffrast.torch.interpolate(
        mesh.get_or_compute_attribute('vertex_normals', should_cache=False),
        rast_out,
        mesh.faces.int(),
        rast_db=rast_out_db,
        diff_attrs='all',
    )[0]
    tangents = nvdiffrast.torch.interpolate(
        mesh.get_or_compute_attribute('vertex_tangents', should_cache=False),
        rast_out,
        mesh.faces.int(),
        rast_db=rast_out_db,
        diff_attrs='all',
    )[0]
    bitangents = torch.nn.functional.normalize(torch.cross(tangents, normals, dim=-1), dim=-1)
    # get uvs
    texc, texd = nvdiffrast.torch.interpolate(
        mesh.uvs,
        rast_out,
        mesh.face_uvs_idx.int(),
        rast_db=rast_out_db,
        diff_attrs='all',
    )
    return normals, tangents, bitangents, texc, texd

def render(mesh, camera, light, random_background=None, val_background=False):
    # transform mesh
    vertices_camera = camera.extrinsics.transform(mesh.vertices)
    vertices_clip = camera.intrinsics.project(vertices_camera)
    faces_int = mesh.faces.int()
    # rasterize
    glctx = dr.RasterizeCudaContext()

    rast_out, rast_out_db = nvdiffrast.torch.rasterize(
        glctx,
        vertices_clip,
        faces_int,
        (camera.height, camera.width),
    )
    rast_out = torch.flip(rast_out, dims=(1,))
    rast_out_db = torch.flip(rast_out_db, dims=(1,))
    mask = torch.clamp(rast_out[..., -1:], 0, 1)
    # interpolate normals, tangents & bitangents
    normals, tangents, bitangents, texc, texd = interpolate_attributes(
        rast_out,
        rast_out_db,
        mesh,
    )
    # texturing
    material = mesh.materials[0]
    def _proc_channel(texture_image):
        if texture_image is None:
            return None
        return nvdiffrast.torch.texture(
            texture_image[None, ...],
            texc,
            texd,
            filter_mode='linear-mipmap-linear',
            # filter_mode='linear', #'linear' 'nearest'
            max_mip_level=9
        )
    mapped_albedo = _proc_channel(material.diffuse_texture)
    mapped_normal = _proc_channel(material.normals_texture)
    mapped_metallic = _proc_channel(material.metallic_texture)
    mapped_roughness = _proc_channel(material.roughness_texture)
    # shading
    if mapped_normal is not None:
        shading_normals = torch.nn.functional.normalize(
            tangents * mapped_normal[..., :1]
            - bitangents * mapped_normal[..., 1:2]
            + normals * mapped_normal[..., 2:3],
            dim=-1,
        )
    else:
        shading_normals = normals
    diffuse_light = light(shading_normals)

    if mapped_metallic is not None and mapped_roughness is not None:
        viewdirs = -get_ray_dirs(camera)
        n_dot_v = (shading_normals * viewdirs).sum(-1, keepdim=True)
        reflective = n_dot_v * shading_normals * 2 - viewdirs

        roughness = torch.clamp(mapped_roughness, min=1e-3)
        specular_light = light(reflective, roughness)
    
        diffuse_albedo = (1 - mapped_metallic) * mapped_albedo
        fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1)
        fg = dr.texture(
            mesh.materials[0].FG_LUT,
            fg_uv.reshape(1, -1, 1, 2).contiguous(),
            filter_mode='linear',
            boundary_mode='clamp',
            ).reshape(*roughness.shape[:-1], 2)
        F0 = (1. - mapped_metallic) * 0.04 + mapped_metallic * mapped_albedo
        specular_albedo = F0 * fg[..., 0:1] + fg[..., 1:2]
        output_image = diffuse_light * diffuse_albedo + specular_light * specular_albedo
    else:
        diffuse_albedo = mapped_albedo
        output_image = diffuse_light * diffuse_albedo

    if val_background:
        background_color = torch.tensor([1, 1, 1], device='cuda:0')
    elif random_background is None:
        background_color = torch.rand(3, device='cuda:0')
    else:
        background_color = random_background


    output_image = torch.where(mask == 1, output_image, background_color)
    return output_image

def get_ray_dirs(camera):
    num_cameras = len(camera)
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(camera.height, device='cuda'),
        torch.arange(camera.width, device='cuda'),
    )
    pixel_x = pixel_x + 0.5
    pixel_x = pixel_x.unsqueeze(0) - camera.x0.view(-1, 1, 1)
    pixel_x = 2 * (pixel_x / camera.width) - 1.0

    pixel_y = pixel_y + 0.5
    pixel_y = pixel_y.unsqueeze(0) - camera.y0.view(-1, 1, 1)
    pixel_y = 2 * (pixel_y / camera.height) - 1.0

    ray_dir = torch.stack((pixel_x * camera.tan_half_fov.view(-1, 1, 1),
                          -pixel_y * camera.tan_half_fov.view(-1, 1, 1),
                          -torch.ones_like(pixel_x)), dim=-1)
    ray_dir = ray_dir.reshape(num_cameras, -1, 3)  # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)
    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir = torch.nn.functional.normalize(ray_dir, dim=-1)
    ray_dir = ray_dir.reshape(-1, camera.height, camera.width, 3)
    return ray_dir