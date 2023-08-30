pip:
	pip install -e imgx
	pip install -e imgx_datasets

test:
	pytest --cov=imgx -n 4 imgx/tests -x
	pytest --cov=imgx_datasets -n 4 imgx_datasets/tests -x

build_dataset:
	tfds build imgx_datasets/imgx_datasets/male_pelvic_mr &
	tfds build imgx_datasets/imgx_datasets/amos_ct &
	tfds build imgx_datasets/imgx_datasets/muscle_us &
	tfds build imgx_datasets/imgx_datasets/brats2021_mr &

rebuild_dataset:
	tfds build imgx_datasets/imgx_datasets/male_pelvic_mr --overwrite &
	tfds build imgx_datasets/imgx_datasets/amos_ct --overwrite &
	tfds build imgx_datasets/imgx_datasets/muscle_us --overwrite &
	tfds build imgx_datasets/imgx_datasets/brats2021_mr --overwrite &
