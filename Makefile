pip:
	pip install -e .

test:
	pytest --cov=imgx -n 4 imgx

build_dataset:
	tfds build imgx/datasets/male_pelvic_mr
	tfds build imgx/datasets/amos_ct
	tfds build imgx/datasets/muscle_us
	tfds build imgx/datasets/brats2021_mr
