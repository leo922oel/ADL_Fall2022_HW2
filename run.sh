# "${1}": path to the context file.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.

python dataset.py $2 ./temp/test.json
python multiple_choice.py ...
python question_answering.py ...