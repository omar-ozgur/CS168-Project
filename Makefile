.DEFAULT_GOAL := run
INPUT=inputs
OUTPUT=outputs
IN_RESIZE=in_resize
OUT_RESIZE=out_resize
COMB=combined
EPOCHS=300
TRAIN=lesions_train
TEST=lesions_test
# Installs any required dependencies
setup:
	pip3 install -r requirements.txt
# Removes old data
clean:
	rm -rf data/$(INPUT) data/$(OUTPUT) data/$(IN_RESIZE) data/$(OUT_RESIZE) data/$(COMB) data/$(TRAIN) data/$(TEST)
# Converts all dicom images in the data dir to pngs
convert:
	python3 tools/mri_to_png.py
# Runs resizing tool to make all images same size
resize:
	python3 tools/process.py --input_dir data/$(INPUT) --operation resize --output_dir data/$(IN_RESIZE)
	python3 tools/process.py --input_dir data/$(OUTPUT) --operation resize --output_dir data/$(OUT_RESIZE)
# Combines input and output images side by side for pix2pix
combine:
	python3 tools/process.py --input_dir data/$(IN_RESIZE) --b_dir data/$(OUT_RESIZE) --operation combine --output_dir data/$(COMB)
# Splits the data into training and testing
split:
	python3 tools/split.py --dir data/$(COMB)
# Runs the pix2pix model on the training data utilizing GPU Tensorflow
train:
	python3 pix2pix.py --mode train --output_dir data/$(TRAIN) --max_epochs $(EPOCHS) --input_dir data/$(COMB)/train --which_direction AtoB
# Runs the trained model on testing data
test:
	python3 pix2pix.py --mode test --output_dir data/$(TEST) --input_dir data/$(COMB)/val --checkpoint data/$(TRAIN)
# Runs pre-processing pipeline
run: clean convert resize combine split
