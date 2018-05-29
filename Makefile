.DEFAULT_GOAL := run
setup:
	pip3 install -r requirements.txt
clean:
	rm -rf data/inputs data/outputs
run: clean
	python3 tools/mri_to_png.py

