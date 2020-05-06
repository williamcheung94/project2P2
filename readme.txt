1.scp proj2 file to /spark-examples/ directory of the master node  
2.change the file_path variable in problem2.py and also the sparksession master to 'spark://$SPARK_MASTER:7077'
3.Run /usr/local/spark/bin/spark-submit /spark-examples/proj2/problem2.py 
4.The script above would print out the accuracy,recall, and precision of the testing result of LR model 
