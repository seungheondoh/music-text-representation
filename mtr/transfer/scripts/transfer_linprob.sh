for DATASET in gtzan fma mtat kvt emotify mtg_top50tags mtg_genre mtg_instrument mtg_moodtheme
do
    python train_probing.py --probe_type linear --batch-size 64 --lr 1e-3 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 64 --lr 1e-3 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 512 --lr 1e-3 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 512 --lr 1e-3 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET

    python train_probing.py --probe_type linear --batch-size 64 --lr 1e-2 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 64 --lr 1e-2 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 512 --lr 1e-2 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 512 --lr 1e-2 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET

    # add dropout
    python train_probing.py --probe_type linear --batch-size 64 --lr 1e-3 --dropout 0.25 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 64 --lr 1e-3 --dropout 0.25 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 512 --lr 1e-3 --dropout 0.25 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 512 --lr 1e-3 --dropout 0.25 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET

    python train_probing.py --probe_type linear --batch-size 64 --lr 1e-2 --dropout 0.25 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 64 --lr 1e-2 --dropout 0.25 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 512 --lr 1e-2 --dropout 0.25 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 512 --lr 1e-2 --dropout 0.25 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET

    # add layer norm
    python train_probing.py --probe_type linear --batch-size 64 --lr 1e-3 --is_norm--framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 64 --lr 1e-3 --is_norm--framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 512 --lr 1e-3 --is_norm--framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 512 --lr 1e-3 --is_norm--framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET

    python train_probing.py --probe_type linear --batch-size 16 --lr 1e-2 --is_norm--framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 16 --lr 1e-2 --is_norm--framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 64 --lr 1e-2 --is_norm--framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 64 --lr 1e-2 --is_norm--framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 512 --lr 1e-2 --is_norm--framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 512 --lr 1e-2 --is_norm--framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET

    python train_probing.py --probe_type linear --batch-size 16 --lr 1e-3 --dropout 0.25 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 16 --lr 1e-3 --dropout 0.25 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 64 --lr 1e-3 --dropout 0.25 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 64 --lr 1e-3 --dropout 0.25 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 512 --lr 1e-3 --dropout 0.25 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 512 --lr 1e-3 --dropout 0.25 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET

    python train_probing.py --probe_type linear --batch-size 64 --lr 1e-2 --dropout 0.25 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 64 --lr 1e-2 --dropout 0.25 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 512 --lr 1e-2 --dropout 0.25 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 512 --lr 1e-2 --dropout 0.25 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET

    python train_probing.py --probe_type linear --batch-size 64 --lr 1e-3 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 64 --lr 1e-3 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 512 --lr 1e-3 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 512 --lr 1e-3 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET

    python train_probing.py --probe_type linear --batch-size 64 --lr 1e-2 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 64 --lr 1e-2 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python train_probing.py --probe_type linear --batch-size 512 --lr 1e-2 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
    python eval_probing.py --probe_type linear --batch-size 512 --lr 1e-2 --is_norm 1 --framework $1 --text_type $2 --text_rep $3 --gpu $4 --eval_dataset $DATASET
done
