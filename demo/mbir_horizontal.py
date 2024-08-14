import numpy as np
import os
import time
import scipy
import mbirjax
import demo_utils
import SimpleITK as sitk
import pprint
import mbirjax.plot_utils as pu
from multipose_utils import transform_utils

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is a demonstration of the metal artifact reduction (MAR) functionality using MAR sinogram weight.\
    \n Demo functionality includes:\
    \n\t * downloading NSI dataset from specified urls;\
    \n\t * Computing sinogram data;\
    \n\t * Computing two sets of sinogram weights, one with type "transmission_root" and the other with type "MAR";\
    \n\t * Computing two sets of MBIR reconstructions with each sinogram weight respectively;\
    \n\t * Displaying the results.\n')
    # ###################### User defined params. Change the parameters below for your own use case.
    output_path = './output/mbir_horizontal/'  # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist

    # Path to NSI scan directory.
    dataset_dir = "/depot/bouman/data/share_conebeam_data/new_MAR_phantom/horiz_metal_with_corrections"

    # #### preprocessing parameters
    downsample_factor = [4, 4]  # downsample factor of scan images along detector rows and detector columns.
    subsample_view_factor = 4  # view subsample factor.

    # #### recon parameters
    sharpness = 0.0
    
    # #### Beam-hardening correction parameter.
    bh_coeff = 0.0 # typical choices are 0.5, 1.0, and 1.5
    # ###################### End of parameters

    print("\n*******************************************************",
          "\n************** NSI dataset preprocessing **************",
          "\n*******************************************************")
    sino, cone_beam_params, optional_params = \
        mbirjax.preprocess.nsi.compute_sino_and_params(dataset_dir,
                                                       downsample_factor=downsample_factor,
                                                       subsample_view_factor=subsample_view_factor)

    # perform beam hardening correction to the sinogram data
    sino = sino + bh_coeff*sino*sino

    print("\n*******************************************************",
          "\n***************** Set up MBIRJAX model ****************",
          "\n*******************************************************")
    # ConeBeamModel constructor
    ct_model = mbirjax.ConeBeamModel(**cone_beam_params)

    # Set additional geometry arguments
    ct_model.set_params(**optional_params)

    # Set image voxel size to be the same as the voxel size from the vertical pose recon
    # this step is to ensure a rigid body transformation
    ct_model.set_params(delta_voxel=0.28700696664031833)
    
    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1, positivity_flag=True)

    # Print out model parameters
    ct_model.print_params()
    print("\n*******************************************************",
          "\n******* Calculate transmission sinogram weights *******",
          "\n*******************************************************")
    weights_trans = ct_model.gen_weights(sino, weight_type='transmission')
    
    print("\n*******************************************************",
          "\n**** Perform recon with transmission_root weights. ****",
          "\n*******************************************************")
    print("This recon will be used to identify metal voxels and compute the MAR sinogram weight.")
    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()

    recon, recon_params = ct_model.recon(sino, weights=weights_trans)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for initial recon is {:.3f} seconds'.format(elapsed))
    np.save(os.path.join(output_path, "recon_horizontal.npy"), recon)
    
    print("\n*******************************************************",
          "\n*********** Perform image transformation. *************",
          "\n*******************************************************")
    print("Load transformation information that maps the image from the horizontal meausurement pose to an upright pose.")
    # The upright pose is defined by manually rotating the vertical pose reconstruction.
    # The transformation information from the horizontal pose to upright pose is estimated using the image registration module in 3D Slicer. 
    # For simplicity, this transformation information is directly provided to the user in a tfm file.
    transform_info_horiz_to_upright = sitk.ReadTransform("registration_info/transform_info_horiz_to_upright.tfm") # load the transformation information from horizontal pose to the upright pose
    
    # apply the transformation to the recon array
    recon_transformed = transform_utils.transformer_sitk(recon, transform_info_horiz_to_upright, output_size=(384,384,480))
    
    print("\n*******************************************************",
          "\n******** Display results in the upright pose. *********",
          "\n*******************************************************")
    # change the image data shape to (slices, rows, cols), so that the rotation axis points up when viewing the coronal/sagittal slices with mbirjax slice_viewer
    recon_transformed = np.transpose(recon_transformed, (2, 1, 0))
    recon_transformed = recon_transformed[:, :, ::-1]

    # ##### display the results in measurement and reconstruction poses respectively
    vmax = downsample_factor[0] * 0.008
    pu.slice_viewer(recon_transformed, vmin=0, vmax=vmax, slice_axis=2, slice_label='Sagittal Slice', title='MBIRJAX recon, upright pose')
    pu.slice_viewer(recon_transformed, vmin=0, vmax=vmax, slice_axis=1, slice_label='Coronal Slice', title='MBIRJAX recon, upright pose')
    pu.slice_viewer(recon_transformed, vmin=0, vmax=vmax, slice_axis=0, slice_label='Axial Slice', title='MBIRJAX recon, upright pose')
    
    print("Done.")
