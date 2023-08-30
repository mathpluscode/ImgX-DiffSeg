"""Dataset module to build tensorflow datasets."""

from imgx_datasets.amos_ct.amos_ct_dataset_builder import AMOS_CT_INFO
from imgx_datasets.brats2021_mr.brats2021_mr_dataset_builder import (
    BRATS2021_MR_INFO,
)
from imgx_datasets.dataset_info import DatasetInfo
from imgx_datasets.male_pelvic_mr.male_pelvic_mr_dataset_builder import (
    MALE_PELVIR_MR_INFO,
)
from imgx_datasets.muscle_us.muscle_us_dataset_builder import MUSCLE_US_INFO

# supported datasets
MALE_PELVIC_MR = "male_pelvic_mr"
AMOS_CT = "amos_ct"
MUSCLE_US = "muscle_us"
BRATS2021_MR = "brats2021_mr"

# dataset info
INFO_MAP: dict[str, DatasetInfo] = {
    MALE_PELVIC_MR: MALE_PELVIR_MR_INFO,
    AMOS_CT: AMOS_CT_INFO,
    MUSCLE_US: MUSCLE_US_INFO,
    BRATS2021_MR: BRATS2021_MR_INFO,
}
