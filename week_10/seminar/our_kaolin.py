from __future__ import annotations
from typing import Sequence, Union, Optional, List
from torch import Tensor
import torch
import copy
import logging
from enum import Enum
from itertools import chain
import os
from PIL import Image
import numpy as np

@torch.jit.script
def rot33_rotate(point: Tensor, mat: Tensor) -> Tensor:
    """Rotate a point using a 3x3 rotation matrix.

    Args:
        point (Tensor): Batch of points to rotate of shape (b, 3).
        mat (Tensor): Batch of 3x3 rotation matrices of shape (b, 3, 3).

    Returns:
        Tensor: Batch of rotated points of shape (b, 3).
    """
    return torch.matmul(mat, point.unsqueeze(-1)).squeeze()  # align batch sizes by appending dummy dimension

def center_points(points: torch.FloatTensor, normalize: bool = False, eps=1e-6):
    r"""Returns points centered at the origin for every pointcloud. If `normalize` is
    set, will also normalize each point cloud spearately to the range of [-0.5, 0.5].
    Note that each point cloud is centered individually.

    Args:
        points (torch.FloatTensor): point clouds of shape :math:`(\text{batch_size}, \text{num_points}, 3)`,
         (other channel numbers supported).
        normalize (bool): if true, will also normalize each point cloud to be in the range [-0.5, 0.5]
        eps (float): eps to use to avoid division by zero when normalizing

    Return:
        (torch.FloatTensor) modified points with same shape, device and dtype as input
    """
    assert len(points.shape) == 3, f'Points have unexpected shape {points.shape}'

    vmin = points.min(dim=1, keepdim=True)[0]
    vmax = points.max(dim=1, keepdim=True)[0]
    vmid = (vmin + vmax) / 2
    res = points - vmid
    if normalize:
        den = (vmax - vmin).max(dim=-1, keepdim=True)[0].clip(min=eps)
        res = res / den
    return res

################################################################################################################
# Materials
################################################################################################################


class MaterialError(Exception):
    pass


class MaterialNotSupportedError(MaterialError):
    pass


class MaterialLoadError(MaterialError):
    pass


class MaterialWriteError(MaterialError):
    pass


class MaterialFileError(MaterialError):
    pass

def load_mtl(mtl_path, error_handler):
    """Load and parse a Material file and return its raw values.

    Followed format described in: https://people.sc.fsu.edu/~jburkardt/data/mtl/mtl.html.
    Currently only support diffuse, ambient and specular parameters (Kd, Ka, Ks)
    through single RGB values or texture maps.

    Args:
        mtl_path (str): Path to the mtl file.

    Returns:
        (dict):
            Dictionary of materials, each a dictionary of properties, containing the following keys, torch.Tensor
            values if present in the mtl file. Only keys present in mtl will be set, capitalization of keys
            will be consistent with original mtl, but both upper and lowercase strings will be parsed.

            - **Kd**: diffuse color of shape (3)
            - **map_Kd**: diffuse texture map of shape (H, W, 3)
            - **Ks**: specular color of shape (3)
            - **map_Ks**: specular texture map of shape (H1, W1, 3)
            - **Ka**: ambient color of shape (3)
            - **map_Ka**: ambient texture map of shape (H2, W2, 3)
            - **bump** or **map_bump**: normals texture, typically of shape (H3, W3, 3)
            - **disp**: displacement map, typically of shape (H3, W3, 1)
            - **map_d**: opacity map, typically of shape (H4, W4, 1)
            - **map_ns**: roughness map
            - **map_refl**: metallic map
            - **material_name**: string name of the material

    Raises:
        MaterialFileError:
            Failed to open material path.
        MaterialLoadError:
            Failed to load material, very often due to path to map_Kd/map_Ka/map_Ks being invalid.
    """
    mtl_data = {}
    root_dir = os.path.dirname(mtl_path)

    def _read_image_with_options(root_dir, data):
        # TOOD: this assumption may be wrong; see https://github.com/tinyobjloader/tinyobjloader/blob/cab4ad7254cbf7eaaafdb73d272f99e92f166df8/models/texture-options-issue-85.mtl#L22
        fpath = data[-1]
        texture_path = os.path.join(root_dir, fpath)
        img = Image.open(texture_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)

        options = {}
        option_vals = []
        for i in range(1, len(data) - 1):
            dval = data[i].strip()
            if dval[0] == '-':
                if len(option_vals) > 0:
                    options[option_vals[0]] = option_vals[1:]
                    option_vals = []
            option_vals.append(dval)

        if len(option_vals) > 0:
            options[option_vals[0]] = option_vals[1:]

        for k, v in options.items():
            if k == '-imfchan':  # parse the channel option
                if len(v) > 0 and len(img.shape) > 2 and img.shape[-1] > 1:
                    if v[0] == 'r':
                        img = img[..., :1]
                    elif v[0] == 'g':
                        img = img[..., 1:2]
                    elif v[0] == 'b':
                        img = img[..., 2:3]
                    else:
                        logging.warning(f'Unrecognized value {v[0]} for flag -imfchan; r, g, or b expected')
            else:
                logging.warning(f'Flag option {k} not supported')
        return torch.from_numpy(img)


    try:
        f = open(mtl_path, 'r', encoding='utf-8')
    except Exception as e:
        error_handler(MaterialFileError(
            f"Failed to load material at path '{mtl_path}':\n{e}"),
            mtl_path=mtl_path, mtl_data=mtl_data)
    else:
        for line in f.readlines():
            data = line.split()
            if len(data) == 0:
                continue
            try:
                if data[0] == 'newmtl':
                    material_name = data[1]
                    mtl_data[material_name] = {'material_name': material_name}
                # TODO: this is not quite right; need to make this agree with standard.
                elif data[0].lower() in {'map_kd', 'map_ka', 'map_ks', 'bump', 'map_bump', 'disp', 'map_d', 'map_ns', 'map_refl'}:
                    mtl_data[material_name][data[0]] = _read_image_with_options(root_dir, data)
                elif data[0].lower() in {'kd', 'ka', 'ks'}:
                    mtl_data[material_name][data[0]] = torch.tensor(
                        [float(val) for val in data[1:]])
            except Exception as e:
                error_handler(MaterialLoadError(
                    f"Failed to load material at path '{mtl_path}':\n{e}"),
                    data=data, mtl_data=mtl_data)
        f.close()
    return mtl_data


def process_materials_and_assignments(materials_dict, material_assignments_dict, error_handler, num_faces,
                                      error_context_str=''):
    """Converts dictionary style materials and assignments to final format (see args/return values).

    Args:
        materials_dict (dict of str to dict): mapping from material name to material parameters
        material_assignments_dict (dict of str to torch.LongTensor): mapping from material name to either
           1) a K x 2 tensor with start and end face indices of the face ranges assigned to that material or
           2) a K, tensor with face indices assigned to that material
        error_handler: handler able to handle MaterialNotFound error - error can be thrown, ignored, or the
            handler can return a dummy material for material not found (if this is not the case, assignments to
            non-existent materials will be lost), e.g. obj.create_missing_materials_error_handler.
        num_faces: total number of faces in the model
        error_context_str (str): any extra info to attach to thrown errors

    Returns:
        (tuple) of:

        - **materials** (list): list of material parameters, sorted alphabetically by their name
        - **material_assignments** (torch.ShortTensor): of shape `(\text{num_faces},)` containing index of the
            material (in the above list) assigned to the corresponding face, or `-1` if no material was assigned.
    """
    def _try_to_set_name(generated_material, material_name):
        if isinstance(generated_material, Mapping):
            generated_material['material_name'] = material_name
        else:
            try:
                generated_material.material_name = material_name
            except Exception as e:
                warnings.warn(f'Cannot set dummy material_name: {e}')

    # Check that all assigned materials exist and if they don't we create a dummy material
    missing_materials = []
    for mat_name in material_assignments_dict.keys():
        if mat_name not in materials_dict:
            dummy_material = error_handler(
                MaterialNotFoundError(f"'Material {mat_name}' not found, but referenced. {error_context_str}"))

            # Either create dummy material or remove assignment
            if dummy_material is not None:
                _try_to_set_name(dummy_material, mat_name)
                materials_dict[mat_name] = dummy_material
            else:
                missing_materials.append(mat_name)

    # Ignore assignments to missing materials (unless handler created dummy material)
    for mat_name in missing_materials:
        del material_assignments_dict[mat_name]

    material_names = sorted(materials_dict.keys())
    materials = [materials_dict[name] for name in material_names]  # Alphabetically ordered materials
    material_assignments = torch.zeros((num_faces,), dtype=torch.int16) - 1

    # Process material assignments to use material indices instead
    for name, values in material_assignments_dict.items():
        mat_idx = material_names.index(name)  # Alphabetically sorted material

        if len(values.shape) == 1:
            indices = values
        else:
            assert len(values.shape) == 2 and values.shape[-1] == 2, \
                f'Unxpected shape {values.shape} for material assignments for material {name} ' \
                f'(expected (K,) or (K, 2)). {error_context_str}'
            # Rewrite (K, 2) tensor of (face_idx_start, face_idx_end] to (M,) tensor of face_idx
            indices = torch.cat(
                [torch.arange(values[r, 0], values[r, 1], dtype=torch.long) for r in range(values.shape[0])])

        # Use face indices as index to set material_id in face-aligned material assignments
        material_assignments[indices] = mat_idx

    return materials, material_assignments

# https://github.com/NVIDIAGameWorks/kaolin/blob/7b3b46efb12ebaf1a339fe54fc14f56643b32f88/kaolin/render/materials.py#L40

def _to_1d_tensor(data):
    if isinstance(data, torch.Tensor):
        return data.reshape(-1).float()
    elif data is None:
        return None
    else:
        return torch.tensor(data).reshape(-1).float()
    
def group_materials_by_name(materials_list, material_assignments):
    def _try_to_get_name(material):
        name = None
        if isinstance(material, Mapping):
            name = material.get('material_name')
        else:
            try:
                name = material.material_name
            except Exception as e:
                warnings.warn(f'Material {type(material)} had no material_name property')
        if name == '':
            name = None
        return name

class Material:
    """Abstract material definition class.
    Defines material inputs and methods to export material properties.
    """
    def __init__(self, name, shader_name):
        self.material_name = str(name)
        self.shader_name = str(shader_name)

class PBRMaterial(Material):

    __value_attributes = [
        'diffuse_color',
        'roughness_value',
        'metallic_value',
        'clearcoat_value',
        'clearcoat_roughness_value',
        'opacity_value',
        'opacity_threshold',
        'ior_value',
        'specular_color',
        'displacement_value',
        'transmittance_value'
    ]

    __texture_attributes = [
        'diffuse_texture',
        'roughness_texture',
        'metallic_texture',
        'clearcoat_texture',
        'clearcoat_roughness_texture',
        'opacity_texture',
        'ior_texture',
        'specular_texture',
        'normals_texture',
        'displacement_texture',
        'transmittance_texture'
    ]

    __colorspace_attributes = [
        'diffuse_colorspace',
        'roughness_colorspace',
        'metallic_colorspace',
        'clearcoat_colorspace',
        'clearcoat_roughness_colorspace',
        'opacity_colorspace',
        'ior_colorspace',
        'specular_colorspace',
        'normals_colorspace',
        'displacement_colorspace',
        'transmittance_colorspace'
    ]

    __misc_attributes = [
        'is_specular_workflow',
        'material_name',
        'shader_name'
    ]

    @classmethod
    def supported_tensor_attributes(cls):
        return cls.__texture_attributes + cls.__value_attributes

    __slots__ = __value_attributes + __texture_attributes + __colorspace_attributes + __misc_attributes

    def __init__(
        self,
        diffuse_color=None,
        roughness_value=None,
        metallic_value=None,
        clearcoat_value=None,
        clearcoat_roughness_value=None,
        opacity_value=None,
        opacity_threshold=None,
        ior_value=None,
        specular_color=None,
        displacement_value=None,
        transmittance_value=None,
        diffuse_texture=None,
        roughness_texture=None,
        metallic_texture=None,
        clearcoat_texture=None,
        clearcoat_roughness_texture=None,
        opacity_texture=None,
        ior_texture=None,
        specular_texture=None,
        normals_texture=None,
        displacement_texture=None,
        transmittance_texture=None,
        is_specular_workflow=False,
        diffuse_colorspace='auto',
        roughness_colorspace='auto',
        metallic_colorspace='auto',
        clearcoat_colorspace='auto',
        clearcoat_roughness_colorspace='auto',
        opacity_colorspace='auto',
        ior_colorspace='auto',
        specular_colorspace='auto',
        normals_colorspace='auto',
        displacement_colorspace='auto',
        transmittance_colorspace='auto',
        material_name='',
        shader_name='UsdPreviewSurface'
    ):
        super().__init__(material_name, shader_name)
        self.diffuse_color = _to_1d_tensor(diffuse_color)
        assert self.diffuse_color is None or self.diffuse_color.shape == (3,)
        self.roughness_value = _to_1d_tensor(roughness_value)
        assert self.roughness_value is None or self.roughness_value.shape == (1,)
        self.metallic_value = _to_1d_tensor(metallic_value)
        assert self.metallic_value is None or self.metallic_value.shape == (1,)
        self.clearcoat_value = _to_1d_tensor(clearcoat_value)
        assert self.clearcoat_value is None or self.clearcoat_value.shape == (1,)
        self.clearcoat_roughness_value = _to_1d_tensor(clearcoat_roughness_value)
        assert self.clearcoat_roughness_value is None or self.clearcoat_roughness_value.shape == (1,)
        self.opacity_value = _to_1d_tensor(opacity_value)
        assert self.opacity_value is None or self.opacity_value.shape == (1,)
        self.opacity_threshold = _to_1d_tensor(opacity_threshold)
        assert self.opacity_threshold is None or self.opacity_threshold.shape == (1,)
        self.ior_value = _to_1d_tensor(ior_value)
        assert self.ior_value is None or self.ior_value.shape == (1,)
        self.specular_color = _to_1d_tensor(specular_color)
        if self.specular_color is not None:
            if self.specular_color.shape == (1,):
                self.specular_color = self.specular_color.repeat(3)
            else:
                assert self.specular_color.shape == (3,)
        self.displacement_value = _to_1d_tensor(displacement_value)
        assert self.displacement_value is None or self.displacement_value.shape == (1,)
        self.transmittance_value = _to_1d_tensor(transmittance_value)
        assert self.transmittance_value is None or self.transmittance_value.shape == (1,)
        assert diffuse_texture is None or diffuse_texture.dim() == 3
        self.diffuse_texture = diffuse_texture
        assert roughness_texture is None or roughness_texture.dim() == 3
        self.roughness_texture = roughness_texture
        assert metallic_texture is None or metallic_texture.dim() == 3
        self.metallic_texture = metallic_texture
        assert clearcoat_texture is None or clearcoat_texture.dim() == 3
        self.clearcoat_texture = clearcoat_texture
        assert clearcoat_roughness_texture is None or clearcoat_roughness_texture.dim() == 3
        self.clearcoat_roughness_texture = clearcoat_roughness_texture
        assert opacity_texture is None or opacity_texture.dim() == 3
        self.opacity_texture = opacity_texture
        assert ior_texture is None or ior_texture.dim() == 3
        self.ior_texture = ior_texture
        assert specular_texture is None or specular_texture.dim() == 3
        self.specular_texture = specular_texture
        assert normals_texture is None or normals_texture.dim() == 3
        self.normals_texture = normals_texture
        assert displacement_texture is None or displacement_texture.dim() == 3
        self.displacement_texture = displacement_texture
        assert transmittance_texture is None or transmittance_texture.dim() == 3
        self.transmittance_texture = transmittance_texture
        assert diffuse_colorspace in ['auto', 'raw', 'sRGB']
        self.diffuse_colorspace = diffuse_colorspace
        assert roughness_colorspace in ['auto', 'raw']
        self.roughness_colorspace = roughness_colorspace
        assert metallic_colorspace in ['auto', 'raw']
        self.metallic_colorspace = metallic_colorspace
        assert clearcoat_colorspace in ['auto', 'raw']
        self.clearcoat_colorspace = clearcoat_colorspace
        assert clearcoat_roughness_colorspace in ['auto', 'raw']
        self.clearcoat_roughness_colorspace = clearcoat_roughness_colorspace
        assert opacity_colorspace in ['auto', 'raw']
        self.opacity_colorspace = opacity_colorspace
        assert ior_colorspace in ['auto', 'raw']
        self.ior_colorspace = ior_colorspace
        assert specular_colorspace in ['auto', 'raw', 'sRGB']
        self.specular_colorspace = specular_colorspace
        assert normals_colorspace in ['auto', 'raw', 'sRGB']
        self.normals_colorspace = normals_colorspace
        self.displacement_colorspace = displacement_colorspace
        assert transmittance_colorspace in ['auto', 'raw']
        self.transmittance_colorspace = transmittance_colorspace
        self.is_specular_workflow = is_specular_workflow

    def write_to_usd(self, file_path, scene_path, bound_prims=None, time=None,
                     texture_dir='', texture_file_prefix='', shader='UsdPreviewSurface'):
        raise DeprecationWarning('PBRMaterial.write_to_usd is deprecated; instead use kaolin.io.usd.export_material')

    def read_from_usd(self, file_path, scene_path, texture_path=None, time=None):
        raise DeprecationWarning('PBRMaterial.read_from_usd is deprecated; instead use kaolin.io.usd.import_material')

    def get_attributes(self, only_tensors=False):
        r"""Returns names of all attributes that are currently set.

        Return:
           (list): list of string names
        """
        res = []
        options = (PBRMaterial.__value_attributes + PBRMaterial.__texture_attributes) if only_tensors else PBRMaterial.__slots__
        for attr in options:
            if getattr(self, attr) is not None:
                res.append(attr)
        return res

    def _construct_apply(self, func, attributes=None):
        r"""Creates a shallow copy of self, applies func() to all (or specified) tensor attributes in the copy,
        for example converting to cuda.
        """
        if attributes is None:
            attributes = self.get_attributes(only_tensors=True)

        my_copy = copy.copy(self)
        for attr in attributes:
            current_val = getattr(my_copy, attr)
            if current_val is not None:
                updated_val = func(current_val)
                setattr(my_copy, attr, updated_val)
        return my_copy

    def to(self, device):
        """Returns a copy where all material attributes that are tensors are put on the provided device.
        Note that behavior of member tensors is consistent with PyTorch ``Tensor.to`` method.

        Arguments:
            device (torch.device): The destination GPU/CPU device.

         Returns:
            (PBRMaterial): The new material.
        """
        return self._construct_apply(lambda t: t.to(device=device))

    def cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        """Returns a copy where all the attributes are on CUDA memory.

        Arguments:
            device (torch.device): The destination GPU device. Defaults to the current CUDA device.
            non_blocking (bool): If True and the source is in pinned memory,
                                 the copy will be asynchronous with respect to the host.
            memory_format (torch.memory_format, optional): the desired memory format of returned Tensor.
                                                           Default: torch.preserve_format.

        Returns:
            (PBRMaterial): The new material.
        """
        return self._construct_apply(
            lambda t: t.cuda(device=device, non_blocking=non_blocking, memory_format=memory_format))

    def cpu(self, memory_format=torch.preserve_format):
        """Returns a copy where all the attributes are on CPU memory.

        Arguments:
            memory_format (torch.memory_format, optional): the desired memory format of returned Tensor.
                                                           Default: torch.preserve_format.

        Returns:
            (PBRMaterial): The new material.
        """
        return self._construct_apply(lambda t: t.cpu(memory_format=memory_format))

    def contiguous(self, memory_format=torch.contiguous_format):
        """Returns a copy where all the attributes are contiguous in memory.

        Arguments:
            memory_format (torch.memory_format, optional): the desired memory format of returned Tensor.
                                                           Default: torch.contiguous_format.

        Returns:
            (PBRMaterial): The new material.
        """
        return self._construct_apply(lambda t: t.contiguous(memory_format=memory_format))

    def hwc(self):
        """Returns a copy where all the image attributes are in HWC layout.

        Returns:
            (PBRMaterial): The new material.
        """
        def _to_hwc(val):
            if val.shape[0] in [1, 3, 4]:
                return val.permute(1, 2, 0)
            return val

        return self._construct_apply(lambda t: _to_hwc(t), PBRMaterial.__texture_attributes)

    def chw(self):
        """Returns a copy where all the image attributes are in CHW layout.

        Returns:
            (PBRMaterial): The new material.
        """
        def _to_chw(val):
            if val.shape[2] in [1, 3, 4]:
                return val.permute(2, 0, 1)
            return val

        return self._construct_apply(lambda t: _to_chw(t), PBRMaterial.__texture_attributes)

    def describe_attribute(self, attr, print_stats=False, detailed=False):
        r"""Outputs an informative string about an attribute; the same method
        used for all attributes in ``to_string``.

         Args:
            print_stats (bool): if to print statistics about values in each tensor
            detailed (bool): if to include additional information about each tensor

        Return:
            (str): multi-line string with attribute information
        """
        assert attr in PBRMaterial.__slots__, f"Unsupported attribute {attr}"
        val = getattr(self, attr)
        res = ''
        if attr in PBRMaterial.__value_attributes:
            res = tensor_info(
                val, name=f'{attr : >33}', print_stats=print_stats, detailed=detailed) + f' {val}'
        elif torch.is_tensor(val):
            res = tensor_info(
                val, name=f'{attr : >33}', print_stats=print_stats, detailed=detailed)
        elif attr in PBRMaterial.__colorspace_attributes:
            if val != "auto":
                res = '{: >33}: {}'.format(attr, val)
        elif val:
            res = '{: >33}: {}'.format(attr, val)
        return res

    def to_string(self, print_stats=False, detailed=False):
        r"""Returns information about attributes as a multi-line string.

        Args:
            print_stats (bool): if to print statistics about values in each tensor
            detailed (bool): if to include additional information about each tensor

        Return:
            (str): multi-line string with attribute information
        """

        res = [f'PBRMaterial object with']
        res.append(self.describe_attribute('material_name'))
        for attr in PBRMaterial.__misc_attributes:
            if attr != 'material_name':
                res.append(self.describe_attribute(attr))
        attributes = self.get_attributes(only_tensors=True)
        for attr in attributes:
            res.append(self.describe_attribute(attr, print_stats=print_stats, detailed=detailed))
        for attr in PBRMaterial.__colorspace_attributes:
            res.append(self.describe_attribute(attr))

        res = [x for x in res if len(x) > 0]
        return '\n'.join(res)

    def __str__(self):
        return self.to_string()

################################################################################################################
# Work with tensors
################################################################################################################

def tensor_info(t, name='', print_stats=False, detailed=False):
    """
    Convenience method to format diagnostic tensor information, including
    shape, type, and optional attributes if specified as string.
    This information can then be logged as:
    logger.debug(tensor_info(my_tensor, 'my tensor'))

    Log output:
    my_tensor: [10, 2, 100, 100] (torch.float32)

    Args:
        t: input pytorch tensor or numpy array or None
        name: human readable name of the tensor (optional)
        print_stats: if True, includes mean/max/min statistics (takes compute time)
        detailed: if True, includes details about tensor properties

    Returns:
        (String) formatted string

    Examples:
        >>> t = torch.Tensor([0., 2., 3.])
        >>> tensor_info(t, 'mytensor', True, True)
        'mytensor: torch.Size([3]) (torch.float32)  - [min 0.0000, max 3.0000, mean 1.6667]  - req_grad=False, is_leaf=True, device=cpu, layout=torch.strided'
    """
    def _get_stats_str():
        if torch.is_tensor(t):
            return ' - [min %0.4f, max %0.4f, mean %0.4f]' % \
                   (torch.min(t).item(),
                    torch.max(t).item(),
                    torch.mean(t.to(torch.float32)).item())
        elif type(t) == np.ndarray:
            return ' - [min %0.4f, max %0.4f, mean %0.4f]' % (np.min(t), np.max(t), np.mean(t))
        else:
            raise RuntimeError('Not implemented for {}'.format(type(t)))

    def _get_details_str():
        if torch.is_tensor(t):
            return ' - req_grad={}, is_leaf={}, layout={}'.format(
                t.requires_grad, t.is_leaf, t.layout)

    if t is None:
        return '%s: None' % name

    if type(t) is dict:
        return '\n'.join([tensor_info(v, name=f'{name}[{k}]:', print_stats=print_stats, detailed=detailed)
                          for k, v in t.items()])

    shape_str = ''
    if hasattr(t, 'shape'):
        shape_str = '%s ' % str(list(t.shape))

    if hasattr(t, 'dtype'):
        type_str = '%s' % str(t.dtype)
    else:
        type_str = '{}'.format(type(t))

    device_str = ''
    if hasattr(t, 'device'):
        device_str = '[{}]'.format(t.device)

    name_str = ''
    if name is not None and len(name) > 0:
        name_str = '%s: ' % name

    return ('%s%s(%s)%s %s %s' %
            (name_str, shape_str, type_str, device_str,
             (_get_stats_str() if print_stats else ''),
             (_get_details_str() if detailed else '')))

def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
    """Check if :class:`torch.Tensor` is valid given set of criteria.

    Args:
        tensor (torch.Tensor): the tensor to be tested.
        shape (list or tuple of int, optional): the expected shape,
            if a dimension is set at ``None`` then it's not verified.
        dtype (torch.dtype, optional): the expected dtype.
        device (torch.device, optional): the expected device.
        throw (bool): if true (default), will throw if checks fail

    Return:
        (bool) True if checks pass
    """
    if shape is not None:
        if len(shape) != tensor.ndim:
            if throw:
                raise ValueError(f"tensor have {tensor.ndim} ndim, should have {len(shape)}")
            return False
        for i, dim in enumerate(shape):
            if dim is not None and tensor.shape[i] != dim:
                if throw:
                    raise ValueError(f"tensor shape is {tensor.shape}, should be {shape}")
                return False
    if dtype is not None and dtype != tensor.dtype:
        if throw:
            raise TypeError(f"tensor dtype is {tensor.dtype}, should be {dtype}")
        return False
    if device is not None and device != tensor.device.type:
        if throw:
            raise TypeError(f"tensor device is {tensor.device.type}, should be {device}")
        return False
    return True

################################################################################################################
# Surface mesh
################################################################################################################

def unindex_vertices_by_faces(face_vertex_features):
    r"""Given per-face per-vertex features, reshapes into flat indexed values and an index buffer
    (necessary for some operations).

    Args:
        face_vertex_features (torch.Tensor): per-face per-vertex features of shape
            :math:`(..., \text{num_faces}, \text{face_size}, \text{num_channels})`, with any number of preceding
            batch dimensions.

    Returns:
        (torch.FloatTensor, torch.LongTensor) with shapes
            :math:`(..., \text{num_faces} * \text{face_size}, \text{num_channels})`, and
            :math:`(\text{num_faces}, \text{face_size})`
    """
    old_shape = list(face_vertex_features.shape)
    nfaces, fsize, nchannels = old_shape[-3:]
    other_dims = old_shape[:-3]
    new_shape = other_dims + [nfaces * fsize] + [nchannels]
    vertex_features = face_vertex_features.reshape(new_shape)
    face_indices = torch.arange(
        0, nfaces * fsize,
        device=face_vertex_features.device,
        dtype=torch.long).reshape((-1, fsize))
    return vertex_features, face_indices

def index_vertices_by_faces(vertices_features, faces):
    r"""Index vertex features to convert per vertex tensor to per vertex per face tensor.

    Args:
        vertices_features (torch.FloatTensor):
            vertices features, of shape
            :math:`(\text{batch_size}, \text{num_points}, \text{knum})`,
            ``knum`` is feature dimension, the features could be xyz position,
            rgb color, or even neural network features.
        faces (torch.LongTensor):
            face index, of shape :math:`(\text{num_faces}, \text{num_vertices})`.
    Returns:
        (torch.FloatTensor):
            the face features, of shape
            :math:`(\text{batch_size}, \text{num_faces}, \text{num_vertices}, \text{knum})`.
    """
    assert vertices_features.ndim == 3, \
        "vertices_features must have 3 dimensions of shape (batch_size, num_points, knum)"
    assert faces.ndim == 2, "faces must have 2 dimensions of shape (num_faces, num_vertices)"
    input = vertices_features.unsqueeze(2).expand(-1, -1, faces.shape[-1], -1)
    indices = faces[None, ..., None].expand(vertices_features.shape[0], -1, -1, vertices_features.shape[-1])
    return torch.gather(input=input, index=indices, dim=1)

def average_face_vertex_features(faces, face_features, num_vertices=None):
    r"""Given features assigned for every vertex of every face, computes per-vertex features by
    averaging values across all faces incident each vertex.

    Args:
       faces (torch.LongTensor): vertex indices of faces of a fixed-topology mesh batch with
            shape :math:`(\text{num_faces}, \text{face_size})`.
       face_features (torch.FloatTensor): any features assigned for every vertex of every face, of shape
            :math:`(\text{batch_size}, \text{num_faces}, \text{face_size}, N)`.
       num_vertices (int, optional): number of vertices V (set to max index in faces, if not set)

    Return:
        (torch.FloatTensor): of shape (B, V, 3)
    """
    if num_vertices is None:
        num_vertices = int(faces.max()) + 1

    B = face_features.shape[0]
    V = num_vertices
    F = faces.shape[0]
    FSz = faces.shape[1]
    Nfeat = face_features.shape[-1]
    vertex_features = torch.zeros((B, V, Nfeat), dtype=face_features.dtype, device=face_features.device)
    counts = torch.zeros((B, V), dtype=face_features.dtype, device=face_features.device)

    faces = faces.unsqueeze(0).repeat(B, 1, 1)
    fake_counts = torch.ones((B, F), dtype=face_features.dtype, device=face_features.device)
    #              B x F          B x F x 3
    # self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    # self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    for i in range(FSz):
        vertex_features.scatter_add_(1, faces[..., i:i + 1].repeat(1, 1, Nfeat), face_features[..., i, :])
        counts.scatter_add_(1, faces[..., i], fake_counts)

    counts = counts.clip(min=1).unsqueeze(-1)
    vertex_normals = vertex_features / counts
    return vertex_normals

def face_normals(face_vertices, unit=False):
    r"""Calculate normals of triangle meshes. Left-hand rule convention is used for picking normal direction.

        Args:
            face_vertices (torch.Tensor):
                of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 3)`.
            unit (bool):
                if true, return normals as unit vectors. Default: False.
        Returns:
            (torch.FloatTensor):
                face normals, of shape :math:`(\text{batch_size}, \text{num_faces}, 3)`
        """
    if face_vertices.shape[-2] != 3:
        raise NotImplementedError("face_normals is only implemented for triangle meshes")
    # Note: Here instead of using the normals from vertexlist2facelist we compute it from scratch
    edges_dist0 = face_vertices[:, :, 1] - face_vertices[:, :, 0]
    edges_dist1 = face_vertices[:, :, 2] - face_vertices[:, :, 0]
    face_normals = torch.cross(edges_dist0, edges_dist1, dim=2)

    if unit:
        face_normals_length = face_normals.norm(dim=2, keepdim=True)
        face_normals = face_normals / (face_normals_length + 1e-10)

    return face_normals

def vertex_tangents(faces, face_vertices, face_uvs, vertex_normals):
    r"""Compute vertex tangents.

    The vertex tangents are useful to apply normal maps during rendering.

    .. seealso::

        https://en.wikipedia.org/wiki/Normal_mapping#Calculating_tangent_space

    Args:
       faces (torch.LongTensor): unbatched triangle mesh faces, of shape
                                 :math:`(\text{num_faces}, 3)`.
       face_vertices (torch.Tensor): unbatched triangle face vertices, of shape
                                     :math:`(\text{num_faces}, 3, 3)`.
       face_uvs (torch.Tensor): unbatched triangle UVs, of shape
                                :math:`(\text{num_faces}, 3, 2)`.
       vertex_normals (torch.Tensor): unbatched vertex normals, of shape
                                      :math:`(\text{num_vertices}, 3)`.

    Returns:
       (torch.Tensor): The vertex tangents, of shape :math:`(\text{num_vertices, 3})`
    """
    # This function is strongly inspired by
    # https://github.com/NVlabs/nvdiffrec/blob/main/render/mesh.py#L203
    tangents = torch.zeros_like(vertex_normals)

    face_uvs0, face_uvs1, face_uvs2 = torch.split(face_uvs, 1, dim=-2)
    fv0, fv1, fv2 = torch.split(face_vertices, 1, dim=-2)
    uve1 = face_uvs1 - face_uvs0
    uve2 = face_uvs2 - face_uvs0
    pe1 = (fv1 - fv0).squeeze(-2)
    pe2 = (fv2 - fv0).squeeze(-2)

    nom = pe1 * uve2[..., 1] - pe2 * uve1[..., 1]
    denom = uve1[..., 0] * uve2[..., 1] - uve1[..., 1] * uve2[..., 0]
    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(
        denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
    )
    vn_idx = torch.split(faces, 1, dim=-1)
    indexing_dim = 0 if face_vertices.ndim == 3 else 1
    # TODO(cfujitsang): optimizable?
    for i in range(3):
        idx = vn_idx[i].repeat(1, 3)
        tangents.scatter_add_(indexing_dim, idx, tang)
    # Normalize and make sure tangent is perpendicular to normal
    tangents = torch.nn.functional.normalize(tangents, dim=1)
    tangents = torch.nn.functional.normalize(
        tangents -
        torch.sum(tangents * vertex_normals, dim=-1, keepdim=True) *
        vertex_normals
    )

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return tangents

logger = logging.getLogger(__name__)

class SurfaceMesh(object):

    class Batching(str, Enum):
        """Batching strategies supported by the ``SurfaceMesh``."""
        # Note: for python>3.11 can use StrEnum instead
        NONE = "NONE"    #: a single unbatched mesh
        FIXED = "FIXED"  #: a batch of meshes with fixed topology (i.e. same faces array)
        LIST = "LIST"    #: a list of meshes of any topologies

    __material_attributes = ['materials']
    __settings_attributes = ['allow_auto_compute',
                             'unset_attributes_return_none']
    __misc_attributes = ['batching'] + __settings_attributes + __material_attributes
    __float_tensor_attributes = [
        'vertices',
        'face_vertices',
        'normals',
        'face_normals',
        'vertex_normals',
        'uvs',
        'face_uvs',
        'vertex_tangents',
        'vertex_colors',
        'vertex_features',
        'face_tangents',
        'face_colors',
        'face_features'
    ]
    __int_tensor_attributes = [
        'faces',
        'face_normals_idx',
        'face_uvs_idx',
        'material_assignments'
    ]
    __indexed_attributes = dict(  # from index : to indexed attribute
        zip(__int_tensor_attributes, ['vertices', 'normals', 'uvs', 'materials']))
    __tensor_attributes = __float_tensor_attributes + __int_tensor_attributes

    # various ways an attribute can be auto-computed, in order of priority
    __computable_attribute_requirements = {
        'vertex_normals': [['faces', 'face_normals']],
        'vertex_tangents': [['faces', 'face_vertices', 'face_uvs', 'vertex_normals']],
        'vertex_colors': [['faces', 'face_colors']],
        'vertex_features': [['faces', 'face_features']],
        'face_vertices': [['faces', 'vertices']],
        'face_normals': [['normals', 'face_normals_idx'], ['vertex_normals', 'faces'], ['vertices', 'faces']],
        'face_uvs': [['uvs', 'face_uvs_idx']],
        'face_tangents': [['faces', 'vertex_tangents']],
        'face_colors': [['faces', 'vertex_colors']],
        'face_features': [['faces', 'vertex_features']]
    }

    # This list is ordered as in the ctor arguments
    # To be used for print related features
    __ordered_tensor_attributes = [
        'vertices',
        'faces',
        'face_vertices',
        'normals',
        'face_normals_idx',
        'face_normals',
        'uvs',
        'face_uvs_idx',
        'face_uvs',
        'vertex_normals',
        'vertex_tangents',
        'vertex_colors',
        'vertex_features',
        'face_tangents',
        'face_colors',
        'face_features',
        'material_assignments'
    ]
    assert set(__ordered_tensor_attributes) == set(__tensor_attributes), \
        "attributes in __ordered_tensor_attributes don't match those in __tensor_attributes: " \
        f"{set(__ordered_tensor_attributes).difference(set(__tensor_attributes))}"

    # Keeping as separate list as things can diverge
    __fixed_topology_attributes = [
        'faces'
    ]

    # This means we cannot set attributes other than these
    __slots__ = __misc_attributes + __tensor_attributes

    @staticmethod
    def supported_tensor_attributes():
        return SurfaceMesh.__ordered_tensor_attributes

    @staticmethod
    def computable_attribute_requirements():
        return SurfaceMesh.__computable_attribute_requirements

    @staticmethod
    def assert_supported(attr):
        if attr not in SurfaceMesh.__slots__:
            raise AttributeError(f'SurfaceMesh does not support attribute named "{attr}"')

    def __init__(self,
                 vertices: Union[torch.FloatTensor, list],
                 faces: Union[torch.LongTensor, list],
                 normals: Optional[Union[torch.FloatTensor, list]] = None,
                 uvs: Optional[Union[torch.FloatTensor, list]] = None,
                 face_uvs_idx: Optional[Union[torch.LongTensor, list]] = None,
                 face_normals_idx: Optional[Union[torch.LongTensor, list]] = None,
                 material_assignments: Optional[Union[torch.Tensor, list]] = None,
                 materials: Optional[list] = None,
                 vertex_normals: Optional[Union[torch.FloatTensor, list]] = None,
                 vertex_tangents: Optional[Union[torch.FloatTensor, list]] = None,
                 vertex_colors: Optional[Union[torch.FloatTensor, list]] = None,
                 vertex_features: Optional[Union[torch.FloatTensor, list]] = None,
                 face_normals: Optional[Union[torch.FloatTensor, list]] = None,
                 face_uvs: Optional[Union[torch.FloatTensor, list]] = None,
                 face_vertices: Optional[Union[torch.FloatTensor, list]] = None,
                 face_tangents: Optional[Union[torch.FloatTensor, list]] = None,
                 face_colors: Optional[Union[torch.FloatTensor, list]] = None,
                 face_features: Optional[Union[torch.FloatTensor, list]] = None,
                 strict_checks: bool = True,
                 unset_attributes_return_none: bool = True,
                 allow_auto_compute: bool = True):

        self.unset_attributes_return_none = unset_attributes_return_none
        self.allow_auto_compute = allow_auto_compute

        assert torch.is_tensor(vertices) or type(vertices) is list, f'unsupported vertices type {type(vertices)}'
        assert torch.is_tensor(faces) or type(faces) is list, f'unsupported faces type {type(faces)}'
        if type(vertices) is list or type(faces) is list or len(faces.shape) == 3:
            batching = SurfaceMesh.Batching.LIST
        elif len(vertices.shape) == 3:
            batching = SurfaceMesh.Batching.FIXED
        else:
            batching = SurfaceMesh.Batching.NONE

        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.uvs = uvs
        self.face_uvs_idx = face_uvs_idx
        self.face_normals_idx = face_normals_idx
        self.material_assignments = material_assignments
        self.materials = materials
        self.vertex_normals = vertex_normals
        self.vertex_tangents = vertex_tangents
        self.vertex_colors = vertex_colors
        self.vertex_features = vertex_features
        self.face_normals = face_normals
        self.face_uvs = face_uvs
        self.face_vertices = face_vertices
        self.face_tangents = face_tangents
        self.face_colors = face_colors
        self.face_features = face_features
        super().__setattr__('batching', batching)

        ok = self.check_sanity()
        if strict_checks and not ok:
            raise ValueError(f'Illegal inputs passed to SurfaceMesh constructor; check log')

    def check_sanity(self):
        attributes = self.get_attributes(only_tensors=True)

        # Set some known values from current attributes
        known_sizes = {'batchsize': len(self)}

        if 'vertices' in attributes and torch.is_tensor(self.vertices) and self.vertices.numel() > 0:
            if self.batching == SurfaceMesh.Batching.NONE:
                known_sizes['numverts'] = self.vertices.shape[0]
            elif self.batching == SurfaceMesh.Batching.FIXED:
                known_sizes['numverts'] = self.vertices.shape[1]
        if 'faces' in attributes and torch.is_tensor(self.vertices) and self.faces.numel() > 0:
            if self.batching in [SurfaceMesh.Batching.NONE, SurfaceMesh.Batching.FIXED]:
                known_sizes['numfaces'] = self.faces.shape[0]
                known_sizes['facesize'] = self.faces.shape[1]

        res = True
        for attr in attributes:
            res = res and SurfaceMesh.__check_attribute(
                attr, getattr(self, attr), self.batching, batchsize=len(self), log_error=True,
                shape=SurfaceMesh.__expected_shape(attr, self.batching, **known_sizes))
        return res

    def is_triangular(self):
        if self.batching in [SurfaceMesh.Batching.NONE, SurfaceMesh.Batching.FIXED]:
            return self.faces.shape[1] == 3
        elif self.batching == SurfaceMesh.Batching.LIST:
            for f in self.faces:
                if f.shape[1] != 3:
                    return False
            return True
        else:
            raise NotImplementedError(f'Unknown batching to determine if mesh is triangular: {self.batching}')

    @classmethod
    def attribute_info_string(cls, batching: SurfaceMesh.Batching):

        def _get_shape(_attr):
            if batching == SurfaceMesh.Batching.LIST:
                return cls.__expected_shape(_attr, batching, batchsize='B', numverts='V_i', numfaces='F_i',
                                            facesize='FSz_i', numnormals='VN_i', numuvs='U_i')
            else:
                return cls.__expected_shape(_attr, batching, batchsize='B', numverts='V', numfaces='F',
                                            facesize='FSz', numnormals='VN', numuvs='U')

        def _format_type(type_str):
            if batching == SurfaceMesh.Batching.LIST:
                return f'[{type_str}]'
            else:
                return f'({type_str})'

        shape_str = 'shapes' if batching == SurfaceMesh.Batching.LIST else 'shape'
        res = [f'Expected SurfaceMesh contents for batching strategy {batching}']
        for attr in cls.__ordered_tensor_attributes:
            if attr in cls.__int_tensor_attributes:
                res.append(f'{attr : >20}: {_format_type("torch.IntTensor")}   of {shape_str} {_get_shape(attr)}')
            if attr in cls.__float_tensor_attributes:
                res.append(f'{attr : >20}: {_format_type("torch.FloatTensor")} of {shape_str} {_get_shape(attr)}')
        for attr in sorted(cls.__material_attributes):
            res.append(f'{attr : >20}: non-tensor attribute')
        return '\n'.join(res)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def describe_attribute(self, attr, print_stats=False, detailed=False):
        SurfaceMesh.assert_supported(attr)

        if not self.has_attribute(attr):
            return 'None'

        val = super().__getattribute__(attr)
        res = ''
        if attr in SurfaceMesh.__ordered_tensor_attributes:
            if self.batching == SurfaceMesh.Batching.LIST or type(val) is list:
                res = '\n'.join([f'{attr : >20}: ['] +
                              [tensor_info(
                                  x, name=f'{idx : >23}', print_stats=print_stats, detailed=detailed)
                               for idx, x in enumerate(val)] + ['{:>23}'.format(']')])
            else:
                res = tensor_info(
                        val, name=f'{attr : >20}', print_stats=print_stats, detailed=detailed)
        elif attr == 'materials':
            if self.batching != SurfaceMesh.Batching.NONE:
                res = '\n'.join(['{: >20}: ['.format('materials')] +
                              [f'{idx : >23}: list of length {len(x)}'
                               for idx, x in enumerate(val)] + ['{:>23}'.format(']')])
            else:
                res = '{: >20}:'.format('materials') + f' list of length {len(val)}'
        else:
            res = '{: >20}: {}'.format(attr, val)
        return res

    def to_string(self, print_stats=False, detailed=False):
        attributes = self.get_attributes(only_tensors=True)
        res = [f'SurfaceMesh object with batching strategy {self.batching}']
        for attr in attributes:
            res.append(self.describe_attribute(attr, print_stats=print_stats, detailed=detailed))

        if self.has_attribute('materials'):
            res.append(self.describe_attribute('materials', print_stats=print_stats, detailed=detailed))

        for attr, req in self._get_computable_attributes().items():
            # req is list of lists of required attribute names
            res.append(
                f'{attr : >20}: if possible, computed on access from: ' +
                ' or '.join(['(' + ', '.join(x) + ')' for x in req]))

        return '\n'.join(res)

    def as_dict(self, only_tensors=False):
        # TODO: add options for usd export
        attr = self.get_attributes(only_tensors=only_tensors)
        return {a: getattr(self, a) for a in attr}

    def get_attributes(self, only_tensors=False):
        res = []
        options = SurfaceMesh.__ordered_tensor_attributes if only_tensors else SurfaceMesh.__slots__
        for attr in options:
            if self.has_attribute(attr):
                res.append(attr)
        return res

    def has_attribute(self, attr: str):
        try:
            super().__getattribute__(attr)
            return True
        except Exception as e:
            return False

    def __deepcopy__(self, memo):
        attr = self.get_attributes()

        kwargs = {a: copy.deepcopy(getattr(self, a), memo) for a in attr}
        del kwargs['batching']
        return SurfaceMesh(**kwargs, strict_checks=False)

    def __copy__(self):
        attr = self.get_attributes()
        kwargs = {}
        for a in attr:
            val = super().__getattribute__(a)
            if torch.is_tensor(val):
                new_val = val
            elif type(val) is list and len(val) > 0 and torch.is_tensor(val[0]):
                new_val = [x for x in val]
            else:
                new_val = copy.copy(val)
            kwargs[a] = new_val
        del kwargs['batching']
        return SurfaceMesh(**kwargs, strict_checks=False)

    def __setattr__(self, attr, value):
        # TODO: Should we add error checks here?
        if attr == 'batching':
            self.set_batching(value)
        elif value is None:
            if self.has_attribute(attr):
                print(f'Deleting {attr}')
                super().__delattr__(attr)
        else:
            super().__setattr__(attr, value)

    def __len__(self):
        if self.batching == SurfaceMesh.Batching.LIST:
            return len(self.vertices)
        elif self.batching == SurfaceMesh.Batching.NONE:
            return 1
        elif self.batching == SurfaceMesh.Batching.FIXED:
            return self.vertices.shape[0]
        else:
            raise NotImplementedError(f'Unsupported batching {self.batching}')

    @staticmethod
    def __expected_shape(name, batching: SurfaceMesh.Batching, batchsize=None,
                         numverts=None, numfaces=None, facesize=None, numnormals=None,
                         numuvs=None):
        B = batchsize
        V = numverts
        VN = numnormals
        F = numfaces
        U = numuvs
        FSz = facesize
        Any = None

        shapes = {'vertices':                      [V, 3],
                  'normals':                       [VN, 3],
                  'uvs':                           [U, 2],
                  'vertex_normals':                [V, 3],
                  'vertex_tangents':               [V, 3],
                  'vertex_colors':                 [V, Any],  # allow RGBA
                  'vertex_features':               [V, Any],
                  'face_normals':                  [F, FSz, 3],
                  'face_uvs':                      [F, FSz, 2],
                  'face_vertices':                 [F, FSz, 3],
                  'face_tangents':                 [F, FSz, 3],
                  'face_colors':                   [F, FSz, Any],  # allow RGBA
                  'face_features':                 [F, FSz, Any],
                  'faces':                         [F, FSz],
                  'material_assignments':          [F],
                  'face_normals_idx':              [F, FSz],
                  'face_uvs_idx':                  [F, FSz]}

        if name not in shapes:
            raise NotImplementedError(f'Cannot get expected shape for attribute {name}')

        if batching in [SurfaceMesh.Batching.NONE, SurfaceMesh.Batching.LIST]:
            return shapes[name]
        elif batching == SurfaceMesh.Batching.FIXED:
            if name in SurfaceMesh.__fixed_topology_attributes:
                return shapes[name]
            else:
                return [B] + shapes[name]
        else:
            raise NotImplementedError(f'Unsupported batching {batching}')

    @staticmethod
    def __check_attribute(name, value, batching, batchsize, log_error=True, **check_tensor_kwargs):
        def _maybe_log(msg):
            if log_error:
                logger.error(msg)

        check_tensor_kwargs['throw'] = False
        if batching == SurfaceMesh.Batching.LIST:
            if type(value) is not list:
                _maybe_log(f'Attribute {name} must have type list for batching type LIST, but got {type(value)}')
                return False
            if len(value) != batchsize:
                _maybe_log(f'Attribute {name} length {len(value)} does not match batchsize {batchsize} '
                           'for batching type LIST')
                return False
            for i, v in enumerate(value):
                if not torch.is_tensor(v):
                    _maybe_log(f'Expected tensor for {name}[i], but got {type(v)}')
                    return False
                if not check_tensor(v, **check_tensor_kwargs):
                    _maybe_log(f'Attribute {name}[i] for batching type LIST has unexpected '
                               f'value {tensor_info(v)} vs {check_tensor_kwargs}')
                    return False
        elif batching in [SurfaceMesh.Batching.NONE, SurfaceMesh.Batching.FIXED]:
            if not torch.is_tensor(value):
                _maybe_log(f'Expected tensor for {name}, but got {type(value)}')
                return False
            if not check_tensor(value, **check_tensor_kwargs):
                _maybe_log(f'Attribute {name} for batching type {batching} has unexpected '
                           f'value {tensor_info(value)} vs {check_tensor_kwargs}')
                return False
        else:
            raise NotImplementedError(f'Unsupported batching {batching}')
        return True

    @staticmethod
    def convert_attribute_batching(val: Union[torch.Tensor, list],
                         from_batching: SurfaceMesh.Batching, to_batching: SurfaceMesh.Batching,
                         is_tensor: bool = True, fixed_topology: bool = False,
                         batch_size: int = None):
        batch_size_guess = None
        if from_batching == SurfaceMesh.Batching.LIST:
            batch_size_guess = len(val)
        elif from_batching == SurfaceMesh.Batching.NONE:
            batch_size_guess = 1
        elif from_batching == SurfaceMesh.Batching.FIXED:
            if is_tensor and not fixed_topology:
                batch_size_guess = val.shape[0]
            elif not is_tensor:
                batch_size_guess = len(val)

        if batch_size is not None:
            if batch_size_guess is not None and batch_size != batch_size_guess:
                raise ValueError(f'Provided batch size {batch_size} disagrees with value {batch_size_guess} guessed from input {val}')
        else:
            if batch_size_guess is None:
                batch_size_guess = 1
            batch_size = batch_size_guess

        if from_batching == to_batching:
            return val
        elif batch_size == 0:
            return val  # TODO: support empty batches
        elif not is_tensor:
            # Material and other non-tensor attributes kept as lists for LIST and FIXED batching
            if to_batching == SurfaceMesh.Batching.NONE:
                if batch_size == 1:
                    val = val[0]
                else:
                    raise ValueError(f'Cannot return unbatched non-tensor attribute from batch of length {batch_size}')
            elif from_batching == SurfaceMesh.Batching.NONE:
                val = [val]
        elif type(val) is list or torch.is_tensor(val):
            if to_batching == SurfaceMesh.Batching.NONE:
                if batch_size != 1:
                    raise ValueError(f'Cannot return unbatched tensor attribute from batch of length {batch_size}')
                if from_batching == SurfaceMesh.Batching.LIST:
                    val = val[0]
                elif from_batching == SurfaceMesh.Batching.FIXED:
                    if not fixed_topology:
                        val = val.squeeze(0)
                else:
                    raise NotImplementedError(f'Unsupported batching {from_batching}')
            elif to_batching == SurfaceMesh.Batching.FIXED:
                if from_batching == SurfaceMesh.Batching.NONE:
                    if not fixed_topology:
                        val = val.unsqueeze(0)
                elif from_batching == SurfaceMesh.Batching.LIST:
                    if fixed_topology:
                        for i in range(1, batch_size):
                            assert torch.allclose(val[0], val[i]), f'Fixed topology attribute must be equivalent for all meshes'
                        val = val[0]
                    else:
                        val = torch.stack(val)
                else:
                    raise NotImplementedError(f'Unsupported_batching {from_batching}')
            elif to_batching == SurfaceMesh.Batching.LIST:
                if from_batching == SurfaceMesh.Batching.NONE:
                    val = [val]
                elif from_batching == SurfaceMesh.Batching.FIXED:
                    if fixed_topology:
                        val = [val for i in range(batch_size)]
                    else:
                        val = [val[i, ...] for i in range(batch_size)]
                else:
                    raise NotImplementedError(f'Unsupported_batching {from_batching}')
            else:
                raise NotImplementedError(f'Unsupported_batching {to_batching}')
        return val

    def getattr_batched(self, attr: str, batching: SurfaceMesh.Batching):
        val = getattr(self, attr)
        is_material = attr in SurfaceMesh.__material_attributes
        is_tensor = attr in SurfaceMesh.__tensor_attributes

        if not is_material and not is_tensor:
            return val

        return SurfaceMesh.convert_attribute_batching(
            val, from_batching=self.batching, to_batching=batching,
            is_tensor=is_tensor, fixed_topology=(attr in SurfaceMesh.__fixed_topology_attributes),
            batch_size=len(self))

    def to_batched(self):
        return self.set_batching(batching=SurfaceMesh.Batching.FIXED)

    def set_batching(self, batching: SurfaceMesh.Batching, skip_errors=False):

        if self.batching == batching:
            return self

        if len(self) == 0:
            return self

        if batching == SurfaceMesh.Batching.NONE and len(self) != 1:
            raise ValueError(f'Cannot create an unbatched mesh from {len(self)} meshes')

        new_attr = {}
        attrs_to_process = self.get_attributes(only_tensors=True) + \
            [x for x in SurfaceMesh.__material_attributes if self.has_attribute(x)]
        for attr in attrs_to_process:
            try:
                val = self.getattr_batched(attr, batching)
            except Exception as e:  # TODO: what's the right error to catch?
                logger.error(f'Failed to convert attribute {attr} with error {e}')
                if skip_errors and attr not in ['vertices', 'faces']:  # required attrs
                    val = None
                else:
                    raise ValueError(f'Cannot convert {attr} to batching {batching} due to: {e}')
            new_attr[attr] = val

        # Set attributes (to avoid messing up internal state while getting attributes in previous loop)
        for attr, val in new_attr.items():
            if val is None:
                delattr(self, attr)
            else:
                setattr(self, attr, val)

        super().__setattr__('batching', batching)
        return self

    @classmethod
    def flatten(cls, meshes: Sequence[SurfaceMesh], skip_errors: bool = False, group_materials_by_name: bool = False):
        mesh = SurfaceMesh.cat(meshes, fixed_topology=False, skip_errors=skip_errors)
        if len(mesh) == 1:
            mesh.set_batching(SurfaceMesh.Batching.NONE)
            return mesh
        _attrs = set(mesh.get_attributes())

        def _informative_cat(values, name, dim=0):
            try:
                return torch.cat(values, dim=dim)
            except Exception as e:
                msg = f'Cannot flatten attribute {name} due to : {e} \n' + \
                    '\n'.join(['Unbatched values are: '] +
                              [tensor_info(x) for x in values])
                if skip_errors:
                    logger.error(msg)
                    return None
                raise ValueError(msg)

        args = {}
        for attr in list(_attrs.intersection(SurfaceMesh.__settings_attributes)):
            args[attr] = getattr(mesh, attr)
            _attrs.remove(attr)

        for attr_index, attr_value in SurfaceMesh.__indexed_attributes.items():
            value_attr_list = None
            index_attr_list = None
            if attr_value in _attrs:
                value_attr_list = getattr(mesh, attr_value)  # e.g. "vertices" or "normals"
                _attrs.remove(attr_value)

            if attr_index in _attrs:
                index_attr_list = getattr(mesh, attr_index)  # e.g. "faces" or "face_normals_idx"
                _attrs.remove(attr_index)
                start_idx = 0
                for i in range(len(index_attr_list)):
                    index_attr_list[i] = index_attr_list[i] + start_idx
                    if value_attr_list is not None:
                        # for tensors, len == shape[0] is guaranteed to be number of values, b/c mesh has LIST batching
                        start_idx += len(value_attr_list[i])
            if index_attr_list is not None:
                args[attr_index] = _informative_cat(index_attr_list, attr_index, dim=0)
            if value_attr_list is not None and len(value_attr_list) > 0:
                if torch.is_tensor(value_attr_list[0]):
                    args[attr_value] = _informative_cat(value_attr_list, attr_value, dim=0)
                else:
                    args[attr_value] = list(chain.from_iterable(value_attr_list))  # list of lists --> list

        for attr in list(_attrs.intersection(SurfaceMesh.__tensor_attributes)):
            args[attr] = torch.cat(getattr(mesh, attr), dim=0)
            _attrs.remove(attr)

        if group_materials_by_name:
            materials = args.get('materials')
            if materials is not None:
                material_assignments = args.get('material_assignments')
                args['materials'], args['material_assignments'] = \
                    group_materials_by_name(materials, material_assignments)

        return SurfaceMesh(**args)

    @classmethod
    def cat(cls, meshes: Sequence[SurfaceMesh], fixed_topology: bool = True, skip_errors: bool = False):
        def _get_joint_attrs():
            # union of all attributes
            attrs_union = set(meshes[0].get_attributes())
            for m in meshes:
                attrs_union.update(set(m.get_attributes()))

            # for every attribute, do all meshes have or can compute it?
            attrs_inter_autocompute = set()
            for att in attrs_union:
                exists_in_all = True
                for m in meshes:
                    if not m.has_or_can_compute_attribute(att):
                        exists_in_all = False
                        break
                if exists_in_all:
                    attrs_inter_autocompute.add(att)

            # intersection of all attributes
            attrs_inter = set(meshes[0].get_attributes())
            for m in meshes:
                attrs_inter.intersection_update(m.get_attributes())

            # for attributes we can compute, do we really need to store it, or is it redundant?
            attrs = copy.deepcopy(attrs_inter_autocompute)
            for att in attrs_inter_autocompute.intersection({'face_normals', 'face_uvs', 'face_vertices'}):
                # All meshes already computed this attribute; we are not doing extra work
                if att in attrs_inter:
                    continue

                priority_requirements = SurfaceMesh.__computable_attribute_requirements[att][0]
                all_met = True
                for req in priority_requirements:
                    for m in meshes:
                        if not m.has_or_can_compute_attribute(req):
                            all_met = False  # this means, would not be able to auto-compute att for concatenated mesh
                            break
                if all_met:
                    attrs.remove(att)

            return attrs

        def _attr_from_meshes(in_attr):
            return list(chain.from_iterable(
                        [m.getattr_batched(in_attr, SurfaceMesh.Batching.LIST) for m in meshes]))

        batchable_attributes = SurfaceMesh.__tensor_attributes + SurfaceMesh.__material_attributes

        if len(meshes) == 0:
            raise ValueError('Zero length list provided to cat operation; at least 1 mesh input required')
        elif len(meshes) == 1:
            res = meshes[0]
        else:
            # Convert all meshes to LIST and create a LIST mesh
            # TODO: this could be more efficient for special cases
            attrs = _get_joint_attrs()
            args = {}
            for attr in attrs:
                if attr in batchable_attributes:
                    args[attr] = _attr_from_meshes(attr)
                elif attr in SurfaceMesh.__settings_attributes:
                    args[attr] = getattr(meshes[0], attr)  # Take first mesh's value

            if fixed_topology:
                # Handle indexed attributes that may not concatenate even for fixed topology meshes
                for indexed_attr in ['normals', 'uvs']:
                    if indexed_attr in args:
                        try:
                            stacked = torch.stack(args[indexed_attr])
                        except Exception as e:
                            logger.warning(f'Cannot cat {indexed_attr} arrays of given shapes; '
                                           f'trying to concatenate face_{indexed_attr} instead, due to: {e}')

                            # Delete indexed attribute and the index that can't be concatenated (e.g. uvs, face_uvs_idx)
                            del args[indexed_attr]
                            face_index_attr = f'face_{indexed_attr}_idx'
                            if face_index_attr in args:
                                del args[face_index_attr]

                            # Auto-compute full attribute value per face instead (this can be concatenated as long as
                            # the number of faces matches)
                            face_attr = f'face_{indexed_attr}'
                            try:
                                args[face_attr] = _attr_from_meshes(face_attr)
                            except Exception as e:
                                logger.warning(f'Cannot compute {face_attr} for all concatenated meshes: {e}')

            res = SurfaceMesh(**args)

        target_batching = SurfaceMesh.Batching.FIXED if fixed_topology else SurfaceMesh.Batching.LIST
        res.set_batching(target_batching, skip_errors=skip_errors)

        return res

    def _requires_grad(self, value):
        res = False
        if torch.is_tensor(value):
            return value.requires_grad
        elif self.batching == SurfaceMesh.Batching.LIST and type(value) is list:
            for v in value:
                if torch.is_tensor(v):
                    res = res or v.requires_grad
                else:
                    logger.warning(f'Unexpected type passed to requires_grad {v}')
        else:
            logger.warning(f'Unexpected type passed to requires_grad {value}')
        return res

    def _uninidex_value_by_faces(self, face_vertex_values):
        can_cache = not self._requires_grad(face_vertex_values)

        if self.batching in [SurfaceMesh.Batching.NONE, SurfaceMesh.Batching.FIXED]:
            val, idx = unindex_vertices_by_faces(face_vertex_values)
            if self.batching == SurfaceMesh.Batching.FIXED:
                idx = idx.unsqueeze(0).tile(face_vertex_values.shape[0], 1, 1)
        elif self.batching == SurfaceMesh.Batching.LIST:
            res = [unindex_vertices_by_faces(x) for x in face_vertex_values]
            val = [x[0] for x in res]
            idx = [x[1] for x in res]
        else:
            raise NotImplementedError(f'Unsupported batching {self.batching}')
        return val, idx, can_cache

    def _index_value_by_faces(self, values, face_idx):
        can_cache = not self._requires_grad(values) and not self._requires_grad(face_idx)
        res = None

        if self.batching == SurfaceMesh.Batching.NONE:
            res = index_vertices_by_faces(values.unsqueeze(0), face_idx).squeeze(0)
        elif self.batching == SurfaceMesh.Batching.FIXED:
            # only faces have fixed topology
            if len(face_idx.shape) == 2:
                # Use fixed topology method
                res = index_vertices_by_faces(values, face_idx)
            else:
                # TODO: add a more flexible index_values_by_face_idx utility
                res = torch.cat([
                    index_vertices_by_faces(values[i:i+1, ...], face_idx[i, ...])
                    for i in range(len(self))], dim=0)
        elif self.batching == SurfaceMesh.Batching.LIST:
            res = [index_vertices_by_faces(values[i].unsqueeze(0), face_idx[i]).squeeze(0)
                   for i in range(len(self))]
        else:
            raise NotImplementedError(f'Unsupported batching {self.batching}')
        return res, can_cache

    def _average_vertex_value_across_faces(self, face_values):

        can_cache = not self._requires_grad(face_values) and not self._requires_grad(self.faces)

        if self.batching == SurfaceMesh.Batching.NONE:
            res = average_face_vertex_features(
                self.faces, face_values.unsqueeze(0), num_vertices=self.vertices.shape[0]).squeeze(0)
        elif self.batching == SurfaceMesh.Batching.FIXED:
            res = average_face_vertex_features(
                self.faces, face_values, num_vertices=self.vertices.shape[1])
        elif self.batching == SurfaceMesh.Batching.LIST:
            res = [average_face_vertex_features(
                self.faces[i], face_values[i].unsqueeze(0), num_vertices=self.vertices[i].shape[0]).squeeze(0)
                   for i in range(len(self))]
        else:
            raise NotImplementedError(f'Unsupported batching {self.batching}')
        return res, can_cache

    def _compute_face_normals_from_vertices(self, should_cache=None):
        args = {'unit': True}
        face_vertices = self.get_or_compute_attribute('face_vertices', should_cache=should_cache)
        can_cache = not self._requires_grad(face_vertices)

        if self.batching == SurfaceMesh.Batching.NONE:
            # When computed this way, there is only one normal per face
            res = face_normals(
                face_vertices.unsqueeze(0), **args).squeeze(0).unsqueeze(1).repeat((1, 3, 1))
        elif self.batching == SurfaceMesh.Batching.FIXED:
            res = face_normals(
                face_vertices, **args).unsqueeze(2).repeat((1, 1, 3, 1))
        elif self.batching == SurfaceMesh.Batching.LIST:
            res = [face_normals(
                face_vertices[i].unsqueeze(0), **args).squeeze(0).unsqueeze(1).repeat((1, 3, 1))
                    for i in range(len(self))]
        else:
            raise NotImplementedError(f'Unsupported batching {self.batching}')
        return res, can_cache

    def _compute_face_uvs(self):
        if self.batching in [SurfaceMesh.Batching.NONE, SurfaceMesh.Batching.FIXED]:
            face_uvs_idx = torch.clone(self.face_uvs_idx)
            face_uvs_idx[face_uvs_idx == -1] = 0
        elif self.batching == SurfaceMesh.Batching.LIST:
            face_uvs_idx = []
            for t in self.face_uvs_idx:
                t = torch.clone(t)
                t[t == -1] = 0
                face_uvs_idx.append(t)
        else:
            raise NotImplementedError(f'Unsupported batching {self.batching}')
        return self._index_value_by_faces(self.uvs, face_uvs_idx)

    def _compute_face_tangents(self, should_cache=None):
        vertex_tangents = self.get_or_compute_attribute('vertex_tangents', should_cache=should_cache)
        return self._index_value_by_faces(vertex_tangents, self.faces)

    def _compute_face_normals(self, should_cache=None):
        if self.has_attribute('normals') and self.has_attribute('face_normals_idx'):
            return self._index_value_by_faces(self.normals, self.face_normals_idx)
        elif self.has_attribute('vertex_normals') and self.has_attribute('faces'):
            return self._index_value_by_faces(self.vertex_normals, self.faces)
        elif self.has_attribute('face_vertices') or (self.has_attribute('vertices') and self.has_attribute('faces')):
            return self._compute_face_normals_from_vertices(should_cache=should_cache)
        else:
            raise RuntimeError(f'This is a bug, _compute_face_normals should never be called if not computable')

    def _compute_vertex_normals(self, should_cache=None):
        # for each vertex, accumulate normal for every face that has it
        face_normals = self.get_or_compute_attribute('face_normals', should_cache=should_cache)
        return self._average_vertex_value_across_faces(face_normals)

    def _compute_vertex_tangents(self, should_cache=None):
        face_vertices = self.get_or_compute_attribute('face_vertices', should_cache=should_cache)
        face_uvs = self.get_or_compute_attribute('face_uvs', should_cache=should_cache)
        vertex_normals = self.get_or_compute_attribute('vertex_normals', should_cache=should_cache)

        can_cache = (
            not self._requires_grad(face_vertices) and
            not self._requires_grad(face_uvs) and
            not self._requires_grad(vertex_normals) and
            not self._requires_grad(self.vertices)
        )

        if self.batching == SurfaceMesh.Batching.NONE:
            res = vertex_tangents(self.faces, face_vertices, face_uvs, vertex_normals)
        elif self.batching == SurfaceMesh.Batching.FIXED:
            res = torch.stack([vertex_tangents(
                self.faces, face_vertices[i, ...], face_uvs[i, ...], vertex_normals[i, ...]) for i in range(len(self))])
        elif self.batching == SurfaceMesh.Batching.LIST:
            res = [vertex_tangents(
                       self.faces[i], face_vertices[i], face_uvs[i], vertex_normals[i]) for i in range(len(self))]
        else:
            raise NotImplementedError(f'Unsupported batching {self.batching}')
        return res, can_cache

    def _compute_computable_attribute(self, attr, should_cache=None):
        if attr in {'face_vertices', 'face_colors', 'face_features'}:  # simple indexing (no recursive compute)
            value_name = 'vertices' if attr == 'face_vertices' else attr.replace('face_', 'vertex_')
            return self._index_value_by_faces(self.get_attribute(value_name), self.faces)
        elif attr in {'vertex_colors', 'vertex_features'}:  # simple averaging (no recursive compute)
            value_name = attr.replace('vertex_', 'face_')
            return self._average_vertex_value_across_faces(self.get_attribute(value_name))
        elif attr == 'vertex_normals':
            return self._compute_vertex_normals(should_cache=should_cache)
        elif attr == 'vertex_tangents':
            return self._compute_vertex_tangents(should_cache=should_cache)
        elif attr == 'face_tangents':
            return self._compute_face_tangents(should_cache=should_cache)
        elif attr == 'face_normals':
            return self._compute_face_normals(should_cache=should_cache)
        elif attr == 'face_uvs':
            return self._compute_face_uvs()
        else:
            logger.error(f'This is a bug; {attr} detected as computable, but computation not implemented')
            return None, False

    def has_or_can_compute_attribute(self, attr: str):
        SurfaceMesh.assert_supported(attr)
        return self._has_or_can_compute_attr(attr)

    def ensure_indexed_attribute(self, attr: str, should_cache: Optional[bool] = None):
        SurfaceMesh.assert_supported(attr)
        if attr not in ['normals', 'uvs']:
            raise ValueError(f'ensure_indexed_attribute only supports "normals" and "uvs"')

        face_val_attr = f'face_{attr}'
        face_idx_attr = f'face_{attr}_idx'
        SurfaceMesh.assert_supported(face_val_attr)  # sanity check code
        SurfaceMesh.assert_supported(face_idx_attr)

        if self.has_attribute(attr) and self.has_attribute(face_idx_attr):
            return self.get_attribute(attr), self.get_attribute(face_idx_attr)  # e.g. normals, face_normals_idx

        # e.g. has face_normals, does not have normals, face_normals_idx
        if self.has_attribute(face_val_attr) and not self.has_attribute(attr) and not self.has_attribute(face_idx_attr):
            val, idx, auto_should_cache = self._uninidex_value_by_faces(self.get_attribute(face_val_attr))
            if should_cache or (should_cache is None and auto_should_cache):
                setattr(self, attr, val)  # e.g. set normals
                setattr(self, face_idx_attr, idx)  # e.g. set face_normals_idx
            return val, idx

        return None, None

    def probably_can_compute_attribute(self, attr: str):
        SurfaceMesh.assert_supported(attr)
        return self._can_compute_attr(attr)[0]

    def _has_or_can_compute_attr(self, attr, allowed_recursion=3):
        if self.has_attribute(attr):
            return True
        return self._can_compute_attr(attr, allowed_recursion=allowed_recursion)[0]

    def _can_compute_attr(self, attr, allowed_recursion=3):
        if allowed_recursion < 0:
            return False, ''

        computable = self._get_computable_attributes()
        if attr not in computable:
            return False, ''

        can_compute = False
        req_str = ''
        for requirements_list in computable[attr]:
            can_compute = True
            for required_attr in requirements_list:
                if not self._has_or_can_compute_attr(required_attr, allowed_recursion=allowed_recursion-1):
                    can_compute = False
                    break
                if attr == 'face_normals' and set(requirements_list) == {'vertices', 'faces'}:
                    can_compute = can_compute and self.is_triangular()
            if can_compute:
                req_str = f'{requirements_list}'
                break
        if not can_compute:
            req_str = ' or '.join(str(x) for x in computable[attr])

        return can_compute, req_str

    def _check_compute_attribute(self, attr, should_cache=None):
        SurfaceMesh.assert_supported(attr)

        throw_info_str = 'To return None instead of throwing, set mesh.unset_attributes_return_none=True'

        # See if we can compute the attribute and issue informative message
        can_compute, req_str = self._can_compute_attr(attr)
        if not can_compute:
            info_str = f'Attribute "{attr}" has not been set and does not have required attributes to be computed: {req_str}'
            if self.unset_attributes_return_none:
                logger.debug(info_str)
                return None
            raise AttributeError(f'{info_str}\n{throw_info_str}')

        # If we can compute, let's compute
        logger.debug(f'Automatically computing {attr} based on {req_str}')
        try:
            computed, auto_should_cache = self._compute_computable_attribute(attr, should_cache=should_cache)
            if should_cache or (should_cache is None and auto_should_cache):
                setattr(self, attr, computed)
            return computed
        except Exception as e:
            info_str = f'Attribute "{attr}" has not been set and failed to be computed due to: {e}'
            if self.unset_attributes_return_none:
                logger.warning(info_str)
                return None
            raise AttributeError(f'{info_str}\n{throw_info_str}')

    def get_or_compute_attribute(self, attr: str, should_cache: Optional[bool] = None):
        if self.has_attribute(attr):
            return getattr(self, attr)

        return self._check_compute_attribute(attr, should_cache=should_cache)

    def get_attribute(self, attr: str):
        if self.has_attribute(attr):
            return getattr(self, attr)

        SurfaceMesh.assert_supported(attr)

        throw_info_str = 'To return None instead of throwing, set mesh.unset_attributes_return_none=True'
        info_str = f'Attribute "{attr}" has not been set'
        if self.unset_attributes_return_none:
            logger.debug(info_str)
            return None
        raise AttributeError(f'{info_str}\n{throw_info_str}')

    def __getattr__(self, attr):
        # Note: this is only called if super().__getattribute__(attr) failed
        SurfaceMesh.assert_supported(attr)

        throw_info_str = 'To return None instead of throwing, set mesh.unset_attributes_return_none=True'

        # If auto-compute disallowed
        if not self.allow_auto_compute:
            info_str = f'Attribute "{attr}" has not been set and allow_auto_compute is off'
            if self.unset_attributes_return_none:
                logger.debug(info_str)
                return None
            raise AttributeError(f'{info_str}\n{throw_info_str}')

        return self._check_compute_attribute(attr)

    def _get_computable_attributes(self):
        exist = self.get_attributes(only_tensors=True)
        computable = {}
        for attr in SurfaceMesh.__ordered_tensor_attributes:
            if attr not in exist and attr in SurfaceMesh.__computable_attribute_requirements:
                computable[attr] = copy.deepcopy(SurfaceMesh.__computable_attribute_requirements[attr])
        return computable

    def cuda(self, device=None, attributes=None):
        return self._construct_apply(lambda t: t.cuda(device), attributes)

    def cpu(self, attributes=None):
        return self._construct_apply(lambda t: t.cpu(), attributes)

    def float_tensors_to(self, float_dtype):
        attributes = set(self.get_attributes(only_tensors=True))
        attributes.intersection_update(SurfaceMesh.__float_tensor_attributes)
        return self._construct_apply(lambda t: t.to(float_dtype), attributes)

    def detach(self, attributes=None):
        return self._construct_apply(lambda t: t.detach(), attributes)

    def _construct_apply(self, func, attributes=None):
        materials_att = 'materials'
        if attributes is None:
            attributes = self.get_attributes(only_tensors=True)
            if self.has_attribute(materials_att):
                attributes.append(materials_att)

        def _construct_apply_material(mat):
            res = mat
            if type(mat) != PBRMaterial:
                logger.warning(f'Mesh material type {type(mat)} is not PBRMaterial; no support for device conversions')
            else:
                try:
                    # We use internal function, allowing mesh extra flexibility for converting PBRMaterial
                    res = mat._construct_apply(func)
                except Exception as e:
                    raise RuntimeError(f'Failed to convert material {type(mat)} using function {func} with error: {e}')
            return res

        my_copy = copy.copy(self)
        for attr in attributes:
            current_val = getattr(my_copy, attr)
            if attr == materials_att:
                try:
                    if self.batching == SurfaceMesh.Batching.NONE:
                        updated_val = [_construct_apply_material(m) for m in self.materials]
                    else:
                        updated_val = [[_construct_apply_material(m) for m in mats] for mats in self.materials]
                    my_copy.materials = updated_val
                except Exception as ex:
                    logger.warning(f'Cannot convert all materials; keeping original: {ex}')
            else:
                if self.batching == SurfaceMesh.Batching.LIST:
                    updated_val = [func(x) for x in current_val]
                else:
                    updated_val = func(current_val)
                setattr(my_copy, attr, updated_val)
        return my_copy

    def to(self, device, attributes=None):
        return self._construct_apply(lambda t: t.to(device), attributes)

    def __getitem__(self, idx):
        if idx > len(self) - 1:
            raise IndexError(f'Out of bound index {idx} for mesh batch of length {len(self)}')

        if self.batching == SurfaceMesh.Batching.NONE:
            return self
        else:
            args = {}
            _attrs = set(self.get_attributes())
            for att in list(_attrs.intersection(set(SurfaceMesh.__settings_attributes))):
                args[att] = self.get_attribute(att)
            for att in _attrs.intersection(set(SurfaceMesh.__material_attributes)):
                args[att] = self.get_attribute(att)[idx]
            for att in _attrs.intersection(set(SurfaceMesh.__tensor_attributes)):
                current_value = self.get_attribute(att)
                if self.batching == SurfaceMesh.Batching.LIST:
                    args[att] = current_value[idx]
                elif self.batching == SurfaceMesh.Batching.FIXED:
                    if att in SurfaceMesh.__fixed_topology_attributes:
                        args[att] = current_value
                    else:
                        args[att] = current_value[idx, ...]
                else:
                    raise NotImplementedError(f'Unsupported batching {self.batching}')
            return SurfaceMesh(**args)

################################################################################################################
# Mesh import
################################################################################################################

def mesh_handler_naive_triangulate(vertices, face_vertex_counts, *features, face_assignments=None):

    def _homogenize(attr, face_vertex_counts):
        if attr is not None:
            attr = attr if isinstance(attr, list) else attr.tolist()
            idx = 0
            new_attr = []
            for face_vertex_count in face_vertex_counts:
                attr_face = attr[idx:(idx + face_vertex_count)]
                idx += face_vertex_count
                while len(attr_face) >= 3:
                    new_attr.append(attr_face[:3])
                    attr_face.pop(1)
            return torch.tensor(new_attr)
        else:
            return None

    def _homogenize_counts(face_vertex_counts, compute_face_id_mappings=False):
        mappings = []  # mappings[i] = [new face ids that i was split into]
        num_faces = 0
        for face_vertex_count in face_vertex_counts:
            attr_face = list(range(0, face_vertex_count))
            new_indices = []
            while len(attr_face) >= 3:
                if compute_face_id_mappings:
                    new_indices.append(num_faces)
                num_faces += 1
                attr_face.pop(1)
            if compute_face_id_mappings:
                mappings.append(new_indices)
        return torch.full((num_faces,), 3, dtype=torch.long), mappings

    new_attrs = [_homogenize(a, face_vertex_counts) for a in features]
    new_counts, face_idx_mappings = _homogenize_counts(face_vertex_counts,
                                                       face_assignments is not None and len(face_assignments) > 0)

    if face_assignments is None:
        # Note: for python > 3.8 can do "return vertices, new_counts, *new_attrs"
        return tuple([vertices, new_counts] + new_attrs)

    # TODO: this is inefficient and could be improved
    new_assignments = {}
    for k, v in face_assignments.items():
        if len(v.shape) == 1:
            new_idx = []
            for old_idx in v:
                new_idx.extend(face_idx_mappings[old_idx])
            new_idx = torch.LongTensor(new_idx)
        else:
            # We support this (start, end] mode for efficiency of OBJ readers
            assert len(v.shape) == 2 and v.shape[1] == 2, 'Expects shape (K,) or (K, 2) for face_assignments'
            new_idx = torch.zeros_like(v)
            for row in range(v.shape[0]):
                old_idx_start = v[row, 0]
                old_idx_end = v[row, 1] - 1
                new_idx[row, 0] = face_idx_mappings[old_idx_start][0]
                new_idx[row, 1] = face_idx_mappings[old_idx_end][-1] + 1
        new_assignments[k] = new_idx

    # Note: for python > 3.8 can do "return vertices, new_counts, *new_attrs, new_assignments"
    return tuple([vertices, new_counts] + new_attrs + [new_assignments])

def import_mesh(path, with_materials=False, with_normals=False,
                error_handler=None, heterogeneous_mesh_handler=None,
                triangulate=False, raw_materials=True):
 
    triangulate_handler = None if not triangulate else utils.mesh_handler_naive_triangulate
    if heterogeneous_mesh_handler is None:
        heterogeneous_mesh_handler = triangulate_handler

    vertices = []
    faces = []
    uvs = []
    # 3 values per face
    face_uvs_idx = []
    normals = []
    # 3 values per face
    face_normals_idx = []

    # materials_dict contains: {material_name: {properties dict}}
    materials_dict = {}

    # material_assignments contain: {material_name: [(face_idx_start, face_idx_end], (face_idx_start, face_idx_end])
    material_assignments_dict = {}
    material_faceidx_start = None
    active_material_name = None

    def _maybe_complete_material_assignment():
        if active_material_name is not None:
            if material_faceidx_start != len(face_uvs_idx):  # Only add if at least one face is assigned
                material_assignments_dict.setdefault(active_material_name, []).append(
                    torch.LongTensor([material_faceidx_start, len(face_uvs_idx)]))

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.split()
            if len(data) == 0:
                continue
            if data[0] == 'v':
                vertices.append(data[1:4])
            elif with_materials and data[0] == 'vt':
                uvs.append(data[1:3])
            elif with_normals and data[0] == 'vn':
                normals.append(data[1:])
            elif data[0] == 'f':
                data = [da.split('/') for da in data[1:]]
                faces.append([int(d[0]) for d in data])
                if with_materials:
                    if len(data[1]) > 1 and data[1][1] != '':
                        face_uvs_idx.append([int(d[1]) for d in data])
                    else:
                        face_uvs_idx.append([0] * len(data))
                if with_normals:
                    if len(data[1]) > 2:
                        face_normals_idx.append([int(d[2]) for d in data])
                    else:
                        face_normals_idx.append([0] * len(data))
            elif with_materials and data[0] == 'usemtl':
                _maybe_complete_material_assignment()
                active_material_name = data[1]
                material_faceidx_start = len(face_uvs_idx)
            elif with_materials and data[0] == 'mtllib':
                mtl_path = os.path.join(os.path.dirname(path), data[1])
                materials_dict.update(load_mtl(mtl_path, error_handler))
    
    # print(uvs)
    # print()
    _maybe_complete_material_assignment()

    vertices = torch.FloatTensor([float(el) for sublist in vertices for el in sublist]).view(-1, 3)
    face_vertex_counts = torch.IntTensor([len(f) for f in faces])
    # key: (Nx2) tensor of (start, end faceidx]
    material_assignments_dict = {k: torch.stack(v) for k, v in material_assignments_dict.items()}

    def _apply_handler(handler):
        all_features = [faces, face_uvs_idx, face_normals_idx]
        # Flatten all features
        all_features = [flatten_feature(f) for f in all_features]
        return handler(vertices, face_vertex_counts, *all_features, face_assignments=material_assignments_dict)

    # Handle non-homogeneous meshes
    is_heterogeneous = face_vertex_counts.numel() > 0 and not torch.all(face_vertex_counts == face_vertex_counts[0])
    if is_heterogeneous:
        if heterogeneous_mesh_handler is None:
            raise utils.NonHomogeneousMeshError(f'Mesh is non-homogeneous '
                                                f'and cannot be imported from {path}.'
                                                f'User can set heterogeneous_mesh_handler.'
                                                f'See kaolin.io.utils for the available options')

        mesh = _apply_handler(heterogeneous_mesh_handler)
        if mesh is None:
            warnings.warn(f'Heterogeneous mesh at path {path} not converted by the handler; returning None.')
            return None
        vertices, face_vertex_counts, faces, face_uvs_idx, face_normals_idx, material_assignments_dict = mesh

    if triangulate_handler is not None and not torch.all(face_vertex_counts == 3):
        mesh = _apply_handler(triangulate_handler)
        if mesh is None:
            warnings.warn(f'Non-triangular mesh at path {path} not triangulated; returning None.')
            return None
        vertices, face_vertex_counts, faces, face_uvs_idx, face_normals_idx, material_assignments_dict = mesh

    faces = torch.LongTensor(faces) - 1

    if with_materials:
        uvs = torch.FloatTensor([float(el) for sublist in uvs
                                 for el in sublist]).view(-1, 2)
        uvs[..., 1] = 1 - uvs[..., 1]
        face_uvs_idx = torch.LongTensor(face_uvs_idx) - 1
        materials, material_assignments = process_materials_and_assignments(
            materials_dict, material_assignments_dict, error_handler, faces.shape[0], error_context_str=path)
        if not raw_materials:
            materials = [raw_material_to_pbr(m) for m in materials]
    else:
        uvs = None
        face_uvs_idx = None
        materials = None
        material_assignments = None

    if with_normals:
        normals = torch.FloatTensor(
            [float(el) for sublist in normals
             for el in sublist]).view(-1, 3)
        face_normals_idx = torch.LongTensor(face_normals_idx) - 1
    else:
        normals = None
        face_normals_idx = None

    return SurfaceMesh(vertices=vertices, faces=faces, uvs=uvs, face_uvs_idx=face_uvs_idx, materials=materials,
                       material_assignments=material_assignments, normals=normals, face_normals_idx=face_normals_idx,
                       unset_attributes_return_none=True)   # for greater backward compatibility
