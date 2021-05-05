import numpy as np
import sys

import torch
import torchio
from torchio.transforms.spatial_transform import SpatialTransform
import SimpleITK as sitk
import nibabel as nib
from torchvision.transforms import Normalize

from GANDLF.utils import resample_image

from torchio.data.subject import Subject
from torchio.transforms.preprocessing.intensity.normalization_transform import NormalizationTransform, TypeMaskingMethod

def normalize_by_val(input_tensor, mean, std):
    """
    This function returns the tensor normalized by these particular values
    """
    normalizer = Normalize(mean, std) 
    return normalizer(input_tensor)             
    
def normalize_imagenet(input_tensor):
    """
    This function returns the tensor normalized by standard imagenet values
    """
    return normalize_by_val(input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                            

def normalize_standardize(input_tensor):
    """
    This function returns the tensor normalized by subtracting 128 and dividing by 128
    image = (image - 128)/128
    """
    return normalize_by_val(input_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    

def normalize_div_by_255(input_tensor):
    """
    This function divides all values of the input tensor by 255 on all channels
    image = image/255
    """
    return normalize_by_val(input_tensor, mean=[0., 0., 0.], std=[1., 1., 1.])

def threshold_intensities(input_tensor, min, max):
    '''
    This function takes an input tensor and 2 thresholds, lower & upper and thresholds between them, basically making intensity values outside this range '0'
    '''
    C = torch.zeros(input_tensor.size(), dtype=input_tensor.dtype)
    l1_tensor = torch.where(input_tensor < max, input_tensor, C)
    l2_tensor = torch.where(l1_tensor > min, l1_tensor, C)
    return l2_tensor


def clip_intensities(input_tensor, min, max):
    '''
    This function takes an input tensor and 2 thresholds, lower and upper and clips between them, basically making the lowest value as 'min' and largest values as 'max'
    '''
    return torch.clamp(input_tensor, min, max)


def resize_image_resolution(input_image, output_size):
    '''
    This function resizes the input image based on the output size and interpolator
    '''
    inputSize = input_image.GetSize()
    outputSpacing = np.array(input_image.GetSpacing())
    for i in range(len(output_size)):
        outputSpacing[i] = outputSpacing[i] * (inputSize[i] / output_size[i])
    return outputSpacing

def apply_resize(input, preprocessing_params, interpolator=sitk.sitkLinear):
    return resample_image(input, resize_image_resolution(input, preprocessing_params['resize']), interpolator=interpolator)

def get_tensor_for_dataloader(input_sitk_image):
    input_image_tensor = torch.from_numpy(sitk.GetArrayFromImage(input_sitk_image)).unsqueeze(0) # single unsqueeze is always needed
    if len(input_image_tensor.shape) == 3: # this is for 2D images
        input_image_tensor = input_image_tensor.unsqueeze(0)
    return input_image_tensor


def tensor_rotate_90(input_image, axis):
    # with 'axis' axis of rotation, rotate 90 degrees
    # tensor image is expected to be of shape (1, a, b, c)
    if axis not in [1, 2, 3]:
        raise ValueError("Axes must be in [1, 2, 3], but was provided as: ", axis)
    relevant_axes = set([1, 2, 3])
    affected_axes = list(relevant_axes - set([axis]) )
    return torch.transpose(input_image, affected_axes[0], affected_axes[1]).flip(affected_axes[1])


def tensor_rotate_180(input_image, axis):
    # with 'axis' axis of rotation, rotate 180 degrees
    # tensor image is expected to be of shape (1, a, b, c)
    if axis not in [1, 2, 3]:
        raise ValueError("Axes must be in [1, 2, 3], but was provided as: ", axis)
    relevant_axes = set([1, 2, 3])
    affected_axes = list(relevant_axes - set([axis]) )
    return input_image.flip(affected_axes[0]).flip(affected_axes[1]) 

# adapted from https://github.com/fepegar/torchio/blob/master/torchio/transforms/preprocessing/intensity/z_normalization.py
class NonZeroNormalizeOnMaskedRegion(NormalizationTransform):
    """Subtract mean and divide by standard deviation.
    Args:
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional keyword arguments.
    """
    def __init__(
            self,
            masking_method: TypeMaskingMethod = None,
            **kwargs
            ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.args_names = ('masking_method',)

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image = subject[image_name]
        mask = image.data != 0
        standardized = self.znorm(
            image.data,
            mask,
        )
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        subject.get_images_dict(intensity_only=True)[image_name]['data'] = standardized

    @staticmethod
    def znorm(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone().float()
        values = tensor.masked_select(mask)
        mean, std = values.mean(), values.std()
        if std == 0:
            return None
        tensor[mask] -= mean
        tensor[mask] /= std
        return tensor

# adapted from https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132933#132933
def crop_image_outside_zeros(array, patch_size):
    dimensions = len(array.shape)
    if dimensions != 4:
        raise ValueError("Array expected to be 4D but got {} dimensions.".format(dimensions)) 
    
    # collapse to single channel and get the mask of non-zero voxels
    mask = array.sum(axis=0) > 0

    # get the small and large corners

    m0 = mask.any(1).any(1)
    m1 = mask.any(0)
    m2 = m1.any(0)
    m1 = m1.any(1)
    
    small = [m0.argmax(), m1.argmax(), m2.argmax()]
    large = [m0[::-1].argmax(), m1[::-1].argmax(), m2[::-1].argmax()]
    large = [m - l for m, l in zip(mask.shape, large)]
    
    # ensure we have a full patch
    # for each axis
    for i in range(3):
        # if less than patch size, extend the small corner out
        if large[i] - small[i] < patch_size[i]:
            small[i] = large[i] - patch_size[i]

        # if bottom fell off array, extend the large corner and set small to 0
        if small[i] < 0:
            small[i] = 0
            large[i] = patch_size[i]

    # calculate pixel location of new bounding box corner (will use to update the reference of the image to physical space)
    new_corner_idxs = np.array([small[0], small[1], small[2]])
    # Get the contents of the bounding box from the array
    new_array = array[:,
                    small[0]:large[0],
                    small[1]:large[1],
                    small[2]:large[2]]
    
    return new_corner_idxs, new_array


# adapted from https://github.com/fepegar/torchio/blob/master/torchio/transforms/preprocessing/spatial/crop.py
class  CropExternalZeroplanes(SpatialTransform):
    """
    Transformation class to enable taking the whole image stack (including segmentation) and removing 
    (starting from edges) physical-coordinate planes with all zero voxels until you reach a non-zero voxel.
    Args:
        patch_size: patch size (used to ensure we do not crop to smaller size than this)
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.args_names = ('patch_size',)
    
    def apply_transform(self, subject):

        # get dictionary of images
        images_dict = subject.get_images_dict(intensity_only=False)
        
        # make sure shapes are consistent across images, and get this shape
        subject.check_consistent_spatial_shape()
        example_image_affine = list(images_dict.values())[0].affine
        
        # create stack of all images (including segmentation)
        numpy_stack_list = []
        names_list = []
        for name, image in images_dict.items():
            numpy_stack_list.append(image.data.numpy().copy())
            names_list.append(name)
        numpy_stack = np.concatenate(numpy_stack_list, axis=0)

        # crop away the external zero-planes on the whole stack
        new_corner_idxs, new_stack = crop_image_outside_zeros(array=numpy_stack, patch_size=self.patch_size)

        # recompute origin of affine matrix using initial image shape
        new_origin = nib.affines.apply_affine(example_image_affine, new_corner_idxs)
        new_affine = example_image_affine.copy()
        new_affine[:3, 3] = new_origin

        # repopulate the subject data and shape
        for idx, array in enumerate(new_stack):
            images_dict[names_list[idx]]['data'] = torch.tensor(np.expand_dims(array, axis=0))
            images_dict[names_list[idx]]['affine'] = new_affine

        return subject

    def is_invertible(self):
        return False
