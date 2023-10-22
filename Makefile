pip:
	pip install -e .

test:
	pytest --cov=imgx -n 4 imgx
	pytest --cov=imgx_datasets -n 4 imgx_datasets

build_dataset:
	tfds build imgx_datasets/male_pelvic_mr
	tfds build imgx_datasets/amos_ct
	tfds build imgx_datasets/muscle_us
	tfds build imgx_datasets/brats2021_mr
