for DATASET in gtzan fma mtat kvt emotify mtg_top50tags mtg_genre mtg_instrument mtg_moodtheme
do
    python eval_zs.py --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
done
