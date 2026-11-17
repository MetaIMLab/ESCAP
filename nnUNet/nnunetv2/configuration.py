import os

from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

default_num_processes = 8 if 'nnUNet_def_n_proc' not in os.environ else int(os.environ['nnUNet_def_n_proc'])
validation_export_num_processes = max(
    1, int(os.environ.get('nnUNet_validation_export_n_proc', '1'))
)

ANISO_THRESHOLD = 3  # determines when a sample is considered anisotropic (3 means that the spacing in the low
# resolution axis must be 3x as large as the next largest spacing)

default_n_proc_DA = get_allowed_n_proc_DA()
