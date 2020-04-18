sets=$1
targetA=$2
targetB=$3
attributeA=$4
attributeB=$5
model_file=$6

source ~/.virtualenvs/bert/bin/activate
export BERT_BASE_DIR=~/workspace/bert/uncased_L-12_H-768_A-12

# Create test data
python create_test_data.py \
  --input_file=./data/sets/${sets}/evals/templates_${targetA}_mask.txt \
  --output_file=./data/gendereval/templates_${targetA}_mask.tfrecord \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt
python create_test_data.py \
  --input_file=./data/sets/${sets}/evals/templates_${targetB}_mask.txt \
  --output_file=./data/gendereval/templates_${targetB}_mask.tfrecord \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt
python create_test_data.py \
  --input_file=./data/sets/${sets}/evals/templates_mask_${attributeA}.txt \
  --output_file=./data/gendereval/templates_mask_${attributeA}.tfrecord \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt
python create_test_data.py \
  --input_file=./data/sets/${sets}/evals/templates_mask_${attributeB}.txt \
  --output_file=./data/gendereval/templates_mask_${attributeB}.tfrecord \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt
python create_test_data.py \
  --input_file=./data/sets/${sets}/evals/templates_mask_mask.txt \
  --output_file=./data/gendereval/templates_mask_mask.tfrecord \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt

# Evaluate test data
python run_pretraining.py \
  --input_file=./data/gendereval/templates_${targetA}_mask.tfrecord \
  --output_dir=${model_file} \
  --eval_file=./data/gendereval/${model_file}/${sets}/templates_${targetA}_mask.pkl \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt
python run_pretraining.py \
  --input_file=./data/gendereval/templates_${targetB}_mask.tfrecord \
  --output_dir=${model_file} \
  --eval_file=./data/gendereval/${model_file}/${sets}/templates_${targetB}_mask.pkl \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt
python run_pretraining.py \
  --input_file=./data/gendereval/templates_mask_${attributeA}.tfrecord \
  --output_dir=${model_file} \
  --eval_file=./data/gendereval/${model_file}/${sets}/templates_mask_${attributeA}.pkl \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt
python run_pretraining.py \
  --input_file=./data/gendereval/templates_mask_${attributeB}.tfrecord \
  --output_dir=${model_file} \
  --eval_file=./data/gendereval/${model_file}/${sets}/templates_mask_${attributeB}.pkl \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt
python run_pretraining.py \
  --input_file=./data/gendereval/templates_mask_mask.tfrecord \
  --output_dir=${model_file} \
  --eval_file=./data/gendereval/${model_file}/${sets}/templates_mask_mask.pkl \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt
