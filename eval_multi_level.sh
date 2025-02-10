device_number=1

python eval_multi_level.py --dataset=cora --device_number=1 > nodecls_outputs/pretraining_outputs/multi_level/cora.txt

python eval_multi_level.py --dataset=citeseer --device_number=1 > nodecls_outputs/pretraining_outputs/multi_level/citeseer.txt

python eval_multi_level.py --dataset=pubmed --device_number=1 > nodecls_outputs/pretraining_outputs/multi_level/pubmed.txt

python eval_multi_level.py --dataset=dblp --device_number=1 > nodecls_outputs/pretraining_outputs/multi_level/dblp.txt

python eval_multi_level.py --dataset=photo --device_number=1 > nodecls_outputs/pretraining_outputs/multi_level/photo.txt

python eval_multi_level.py --dataset=computers --device_number=1 >> nodecls_outputs/pretraining_outputs/multi_level/computers.txt