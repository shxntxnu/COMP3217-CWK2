setup:
	pip3 install -r requirements.txt

run_part_a:
	python3 part_a.py TrainingDataBinary.csv TestingDataBinary.csv TestingResultsBinary.csv

run_part_b:
	python3 part_b.py TrainingDataMulti.csv TestingDataMulti.csv TestingResultsMulti.csv
