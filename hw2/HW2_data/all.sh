for i in `seq 2 20`;
    do
    	echo $i clusters
    	echo KPP
    	echo tf
    	python eval.py output/kpp_HW2_dev_tf_${i}_document_file.txt HW2_dev.gold_standards
    	echo KMEANS
    	echo tf
    	python eval.py output/kmeans_HW2_dev_tf_${i}_document_file.txt HW2_dev.gold_standards
    done    