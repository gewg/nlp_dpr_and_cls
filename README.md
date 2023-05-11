# nlp_dpr_and_cls

nohup python evidence_retriever/main.py --train >evidence_retriever/out/train.out 2>&1 &
nohup python evidence_retriever/main.py --retrain >evidence_retriever/out/train.out 2>&1 &

nohup python classifier/main.py --train >classifier/out/train.out 2>&1 &

python evidence_retriever/main.py --predict
python classifier/main.py --predict