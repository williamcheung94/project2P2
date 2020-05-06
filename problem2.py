# Import the PySpark module
from pyspark.sql import SparkSession
# Create SparkSession object
#/usr/local/spark/bin/pyspark
spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('test') \
                    .getOrCreate()
#master spark://13.59.151.161:7077
# Import the PySpark module
file_path='file:framingham.csv'
data = spark.read.csv(file_path,
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')
data.show(10)
data.count()
data.printSchema()
#data.filter('(male|age) IS NULL').count()
data_dropna=data.dropna()
print("Total number of rows with missing values:",data.count()-data_dropna.count())
print('Exclude missing values')


from pyspark.ml.feature import VectorAssembler
# Create an assembler object
assembler = VectorAssembler(inputCols=['male', 'age' ,'education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose'], outputCol='features')
# Consolidate predictor columns
data_assembled = assembler.transform(data_dropna)
data_assembled = data_assembled.withColumn('label', data_assembled.TenYearCHD.cast('integer'))

# Split into training and testing sets in a 80:20 ratio
train, test = data_assembled.randomSplit([0.8,0.2],seed=17)
# Check that training set has around 80% of records
training_ratio = train.count() / data_assembled.count()
print('training_data_ratio:',training_ratio)

# Import the logistic regression class
from pyspark.ml.classification import LogisticRegression
# Create a classifier object and train on training data
logistic = LogisticRegression().fit(train)
# Create predictions for the testing data and show confusion matrix
prediction = logistic.transform(test)
prediction.groupBy('label', 'prediction').count().show()
# Calculate the elements of the confusion matrix
TN = prediction.filter('prediction = 0 AND label = prediction').count()
TP = prediction.filter('prediction = 1 AND label = prediction').count()
FN = prediction.filter('prediction = 0 AND label != prediction').count()
FP = prediction.filter('prediction = 1 AND label != prediction').count()

# Accuracy measures the proportion of correct predictions
accuracy = float(TP+TN)/float(TP+TN+FP+FN)
Precision = float(TP)/float(TP+FP)
Recall = float(TP)/float(TP+FN)

print("testing data Accuracy:", accuracy)
print("testing data Recall:", Recall)
print("testing data Precision:",Precision )

#from pyspark.ml.feature import StringIndexer
##label column and categorical data 
# Create an indexer
#indexer = StringIndexer(inputCol='carrier', outputCol='carrier_idx')
# Indexer identifies categories in the data
#indexer_model = indexer.fit(data_dropna)
# Indexer creates a new column with numeric index values
#data_indexed = indexer_model.transform(data_dropna)
