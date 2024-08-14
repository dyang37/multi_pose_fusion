import SimpleITK as sitk
import numpy as np
import time
from scipy.ndimage import rotate
import copy
import warnings
import os

def sitk_array_to_image(image_array, pixel_pitch=1.0, origin=None):
    image_sitk = sitk.GetImageFromArray(np.swapaxes(image_array,0,2))
    image_sitk.SetSpacing([pixel_pitch for _ in range(len(image_sitk.GetSpacing()))])
    if origin is None:
        image_sitk.SetOrigin(-(np.array(image_sitk.GetSize())-1)*pixel_pitch/2)
    else:
        image_sitk.SetOrigin(origin)
    return image_sitk

def sitk_image_to_array(image_sitk):
    image_array = sitk.GetArrayFromImage(image_sitk)
    image_array =  np.swapaxes(image_array,0,2)
    return image_array

def transformer_sitk(moving_image, transform_params, fixed_image=None, pixel_pitch=1.0, pixel_pitch_ref=1.0, origin=None, cval=0.0, interp_method='bspline'):
    
    # convert numpy array to sitk image object
    moving_image_sitk = sitk_array_to_image(moving_image, pixel_pitch=pixel_pitch, origin=origin)
    
    # BSpline interpolation
    interpolator = sitk.sitkBSpline

    if fixed_image is None:
        fixed_image = moving_image    
    
    # resampling with fixed image
    fixed_image_sitk = sitk_array_to_image(fixed_image, pixel_pitch=pixel_pitch_ref, origin=origin)
    print("fixed image. Origin = ", fixed_image_sitk.GetOrigin(), "  Spacing = ", fixed_image_sitk.GetSpacing(), " Dimensions = ", fixed_image_sitk.GetDimension(), "Direction = ", fixed_image_sitk.GetDirection())
    print("moving image. Origin = ", moving_image_sitk.GetOrigin(), "  Spacing = ", moving_image_sitk.GetSpacing(), " Dimensions = ", moving_image_sitk.GetDimension(), "Direction = ", moving_image_sitk.GetDirection())
    moving_image_transformed_sitk = sitk.Resample(moving_image_sitk,
                                                  fixed_image_sitk,
                                                  transform_params,
                                                  interpolator,
                                                  cval,
                                                  moving_image_sitk.GetPixelID())
    
    return sitk_image_to_array(moving_image_transformed_sitk)
