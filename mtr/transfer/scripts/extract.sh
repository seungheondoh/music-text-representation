for DATASET in gtzan fma mtat kvt emotify mtg
do
    python extractor.py --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
done
