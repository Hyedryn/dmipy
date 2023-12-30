import os
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs

noddi_path = "output"
lambda_iso_diff=3.e-9
lambda_par_diff=1.7e-9
use_amico=False
core_count=1

# initialize the compartments model
from dmipy.signal_models import cylinder_models, gaussian_models
ball = gaussian_models.G1Ball()
stick = cylinder_models.C1Stick()
zeppelin = gaussian_models.G2Zeppelin()

# watson distribution of stick and Zepelin
from dmipy.distributions.distribute_models import SD1WatsonDistributed
watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp', 'C1Stick_1_lambda_par',
                                               'partial_volume_0')
watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', lambda_par_diff)

# build the NODDI model
from dmipy.core.modeling_framework import MultiCompartmentModel
NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])

# fix the isotropic diffusivity
NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', lambda_iso_diff)

# load the data
data, affine = load_nifti("data/sub-AllTR_dmri_preproc.nii.gz")
bvals, bvecs = read_bvals_bvecs("data/sub-AllTR_dmri_preproc.bval", "data/sub-AllTR_dmri_preproc.bvec")


# load the mask
mask_path = "data/sub-AllTR_brain_mask.nii.gz"
if os.path.isfile(mask_path):
    mask, _ = load_nifti(mask_path)
else:
    mask = np.ones(data.shape[:-1], dtype=bool)

# transform the bval, bvecs in a form suited for NODDI
from dipy.core.gradients import gradient_table
from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
b0_threshold = np.min(bvals) + 10
b0_threshold = max(50, b0_threshold)
gtab_dipy = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
acq_scheme_dmipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_threshold*1e6)

# Reduce mask true area to speed up testing:
mask[0:100, 0:80, :] = False
print("Number of voxels: ", np.sum(mask))
print("Number of total voxels: ", np.prod(mask.shape), " (", mask.shape, ")")
print("Percentage of voxels to be processed: ", np.sum(mask) / np.prod(mask.shape) * 100)

if use_amico:
    # fit the model to the data using noddi amico
    from dmipy.optimizers import amico_cvxpy
    NODDI_fit = amico_cvxpy.AmicoCvxpyOptimizer(acq_scheme_dmipy, data, mask=mask)
else:
    # fit the model to the data
    NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, data, mask=mask,use_parallel_processing=bool(core_count>1),number_of_processors=core_count)
    # NODDI_fit = NODDI_mod.fit(acq_scheme_dmipy, data, mask=mask, solver='mix', maxiter=300)

# exctract the metrics
fitted_parameters = NODDI_fit.fitted_parameters
mu = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_mu"]
odi = fitted_parameters["SD1WatsonDistributed_1_SD1Watson_1_odi"]
f_iso = fitted_parameters["partial_volume_0"]
f_bundle = fitted_parameters["partial_volume_1"]
f_intra = (fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1'])
f_icvf = fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * (fitted_parameters['partial_volume_1']>0.05)
f_extra = ((1 - fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) * fitted_parameters[
    'partial_volume_1'])
mse = NODDI_fit.mean_squared_error(data)
R2 = NODDI_fit.R2_coefficient_of_determination(data)

noddi_path = "output"
patient_path = "sub-AllTR"
# save the nifti
save_nifti(noddi_path + '/' + patient_path + '_noddi_mu.nii.gz', mu.astype(np.float32), affine)
save_nifti(noddi_path + '/' + patient_path + '_noddi_odi.nii.gz', odi.astype(np.float32), affine)
save_nifti(noddi_path + '/' + patient_path + '_noddi_fiso.nii.gz', f_iso.astype(np.float32), affine)
save_nifti(noddi_path + '/' + patient_path + '_noddi_fbundle.nii.gz', f_bundle.astype(np.float32), affine)
save_nifti(noddi_path + '/' + patient_path + '_noddi_fintra.nii.gz', f_intra.astype(np.float32), affine)
save_nifti(noddi_path + '/' + patient_path + '_noddi_icvf.nii.gz', f_icvf.astype(np.float32), affine)
save_nifti(noddi_path + '/' + patient_path + '_noddi_fextra.nii.gz', f_extra.astype(np.float32), affine)
save_nifti(noddi_path + '/' + patient_path + '_noddi_mse.nii.gz', mse.astype(np.float32), affine)
save_nifti(noddi_path + '/' + patient_path + '_noddi_R2.nii.gz', R2.astype(np.float32), affine)
