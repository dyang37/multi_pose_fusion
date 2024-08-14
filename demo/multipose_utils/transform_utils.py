import SimpleITK as sitk
import numpy as np
import time
from scipy.ndimage import rotate
import copy
import warnings
import os

def sitk_array_to_image(image_array, pixel_pitch=1.0):
    image_sitk = sitk.GetImageFromArray(np.swapaxes(image_array,0,2))
    image_sitk.SetSpacing([pixel_pitch for _ in range(len(image_sitk.GetSpacing()))])
    image_sitk.SetOrigin(-(np.array(image_sitk.GetSize())-1)*pixel_pitch/2)
    return image_sitk

def sitk_image_to_array(image_sitk):
    image_array = sitk.GetArrayFromImage(image_sitk)
    image_array =  np.swapaxes(image_array,0,2)
    return image_array

def transformer_sitk(moving_image, transform_params, output_size=None, pixel_pitch=1.0, default_pixel_val=0.0, interp_method='bspline'):
    
    # convert numpy array to sitk image object
    moving_image_sitk = sitk_array_to_image(moving_image, pixel_pitch=pixel_pitch)
    
    # BSpline interpolation
    interpolator = sitk.sitkBSpline

    ### Define size, origin, and pixel pitch for output image
    # if output size is not specified, then set output size the same as input size
    if output_size is None:
        output_size = moving_image_sitk.GetSize()

    output_origin = -(np.array(output_size)-1)*pixel_pitch/2
    output_spacing = moving_image_sitk.GetSpacing() 
 
    # resampling with fixed image
    print("moving image. Origin = ", moving_image_sitk.GetOrigin(), "  Spacing = ", moving_image_sitk.GetSpacing(), " Dimensions = ", moving_image_sitk.GetDimension(), "Direction = ", moving_image_sitk.GetDirection())
    moving_image_transformed_sitk = sitk.Resample(moving_image_sitk,
                                                  size=output_size,
                                                  transform=transform_params,
                                                  interpolator=interpolator,
                                                  outputOrigin=output_origin,
                                                  outputSpacing=output_spacing,
                                                  defaultPixelValue=default_pixel_val,
                                                  outputPixelType=moving_image_sitk.GetPixelID())
    
    return sitk_image_to_array(moving_image_transformed_sitk)
