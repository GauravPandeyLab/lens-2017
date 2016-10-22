/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].
*/

import java.io.*
import java.text.*
import java.util.*
import java.util.zip.*
import weka.classifiers.*
import weka.classifiers.meta.*
import weka.core.*
import weka.core.converters.ConverterUtils.DataSource
import weka.filters.*
import weka.filters.supervised.instance.*
import weka.filters.unsupervised.attribute.*
import weka.filters.unsupervised.instance.*
import Utilities 

/** groovy pipeline.groovy ../diab 0 0 0 weka.classifiers.bayes.NaiveBayes -D **/  

Utilities u = new Utilities()
Instances balance(instances) 
{
    balanceFilter = new SpreadSubsample()
    balanceFilter.setDistributionSpread(1.0)
    balanceFilter.setInputFormat(instances)
    return Filter.useFilter(instances, balanceFilter)
}


// parse args
rootDir 	= args[0]
int bag         = Integer.valueOf(args[1])
int seed 	= Integer.valueOf(args[2])
currentFold 	= args[3]
String[] classifierString = args[4..-1]


// process classifier options
String classifierName = classifierString[0]
String shortClassifierName = classifierName.split("\\.")[-1]
String[] classifierOptions = new String[0]
if (classifierString.length > 1) 
{
    classifierOptions = classifierString[1..-1]
}


// load data parameters from properties file
p = new Properties()
p.load(new FileInputStream(args[0] + "/config.txt"))
inputFilename       = p.getProperty("inputFilename").trim()
classifierDir	    = p.getProperty("classifierDir", ".").trim()
idAttribute         = p.getProperty("idAttribute", "").trim()
classAttribute      = p.getProperty("classAttribute").trim()
balanceTraining     = Boolean.valueOf(p.getProperty("balanceTraining", "false"))
balanceTest         = Boolean.valueOf(p.getProperty("balanceTest", "false"))
balanceValidation   = Boolean.valueOf(p.getProperty("balanceValidation", "false"))
assert p.containsKey("foldCount") || p.containsKey("foldAttribute")
if (p.containsKey("foldCount")) 
{
    foldCount       = Integer.valueOf(p.getProperty("foldCount"))
}


// creating the classifier directory
classifierDirectory = new File(classifierDir, classifierName)
if( !classifierDirectory.exists() ) 
{
    classifierDirectory.mkdirs()
}


// name of the ARFF attribute containing values for leave-one-value-out cross validation. This or foldCount must be specified.
foldAttribute       = p.getProperty("foldAttribute", "").trim()
writeModel          = Boolean.valueOf(p.getProperty("writeModel", "false"))



// load data, determine if regression or classification
source              = new DataSource(inputFilename) 
data                = source.getDataSet()
regression          = data.attribute(classAttribute).isNumeric()
if (!regression) 
{
    predictClassValue = p.getProperty("predictClassValue").trim()
    otherClassValue = p.getProperty("otherClassValue").trim()
}



// shuffle data, set class variable
data.setClass(data.attribute(classAttribute))
data.randomize(new Random(seed))
if (!regression) 
{
    predictClassIndex = data.attribute(classAttribute).indexOfValue(predictClassValue)
    assert predictClassIndex != -1
    //printf " - class distribution in  data: %d:%d=%f\n", u.numWithClass(data, predictClassValue), u.numWithClass(data, otherClassValue),  u.numWithClass(data, predictClassValue)/u.numWithClass(data, otherClassValue)
} 
else 
{
    //printf "[%s] %s, generating predictions\n", shortClassifierName, data.attribute(classAttribute)
}


// add ids if not specified
if (idAttribute == "") 
{
    idAttribute = "ID"
    idFilter = new AddID()
    idFilter.setIDIndex("last")
    idFilter.setInputFormat(data)
    data = Filter.useFilter(data, idFilter)
}


// generate folds
if (foldAttribute != "") 
{
    foldCount = data.attribute(foldAttribute).numValues()
    print "foldCount %r", foldCount
    foldAttributeIndex = String.valueOf(data.attribute(foldAttribute).index() + 1) // 1-indexed
    foldAttributeValueIndex = String.valueOf(data.attribute(foldAttribute).indexOfValue(currentFold) + 1) // 1-indexed
    //printf "[%s] generating %s folds for leave-one-value-out CV\n", shortClassifierName, foldCount

    testFoldFilter = new RemoveWithValues()
    testFoldFilter.setModifyHeader(false)
    testFoldFilter.setAttributeIndex(foldAttributeIndex)
    testFoldFilter.setNominalIndices(foldAttributeValueIndex)
    testFoldFilter.setInvertSelection(true)
    testFoldFilter.setInputFormat(data)
    test = Filter.useFilter(data, testFoldFilter)

    trainingFoldFilter = new RemoveWithValues()
    trainingFoldFilter.setModifyHeader(false)
    trainingFoldFilter.setAttributeIndex(foldAttributeIndex)
    trainingFoldFilter.setNominalIndices(foldAttributeValueIndex)
    trainingFoldFilter.setInvertSelection(false)
    trainingFoldFilter.setInputFormat(data)
    train = Filter.useFilter(data, trainingFoldFilter)
} 
else 
{
    //printf "[%s] generating folds for %s-fold CV\n", shortClassifierName, foldCount
    /* https://weka.wikispaces.com/Generating+cross-validation+folds+(Java+approach)
       stratification is mandatory for ensuring that the class distribution remains 
       the same as in the original dataset (? apparently missing from datasink ?) */
    
    data.stratify(foldCount)
    test = data.testCV(foldCount, Integer.valueOf(currentFold))
    train_all = data.trainCV(foldCount, Integer.valueOf(currentFold), new Random(seed))
    Instances[] subsets = u.getFractions(train_all, 0.25, 0.75)
    train = new Instances(subsets[1])
    validation = new Instances(subsets[0])
}

train = train.resample(new Random(bag), )

//balance if required
if (!regression && balanceTraining) 
{
    //printf "[%s] balancing training samples\n", shortClassifierName
    train = balance(train)
}
if (!regression && balanceTest) 
{
    //printf "[%s] balancing test samples\n", shortClassifierName
    test = balance(test)
}
if (!regression && balanceValidation)
{	
    //printf "[%s] balancing validation\n", shortClassifierName
    validation = balance(validation)
}


// init filtered classifier
classifier = AbstractClassifier.forName(classifierName, classifierOptions)
removeFilter = new Remove()
if (foldAttribute != "") 
{
    removeIndices = new int[2]
    removeIndices[0] = data.attribute(foldAttribute).index()
    removeIndices[1] = data.attribute(idAttribute).index()
} 
else 
{
    removeIndices = new int[1]
    removeIndices[0] = data.attribute(idAttribute).index()
}
removeFilter.setAttributeIndicesArray(removeIndices)
filteredClassifier = new FilteredClassifier()
filteredClassifier.setClassifier(classifier)
filteredClassifier.setFilter(removeFilter)


// train, store duration
start = System.currentTimeMillis()
filteredClassifier.buildClassifier(train)
duration = System.currentTimeMillis() - start
durationMinutes = duration / (1e3 * 60)
//printf "[%s] trained in %.2f minutes, evaluating . . .", shortClassifierName, durationMinutes
header = sprintf "# [%s] %.2f minutes (%d instances, %d:%d=%f)\nid,label,prediction\n", shortClassifierName, durationMinutes, train.numInstances(), u.numWithClass(train, predictClassValue), u.numWithClass(train, otherClassValue),  u.numWithClass(train, predictClassValue)/u.numWithClass(train, otherClassValue)



// write predictions to csv
if (!classifierDirectory.exists()) 
{
    classifierDirectory.mkdir()
}


//write model to file
outputPrefix = sprintf "b%d-f%s-s%d", bag, currentFold, seed
if (writeModel) 
{
    SerializationHelper.write(new GZIPOutputStream(new FileOutputStream(new File(classifierDirectory, shortClassifierName + "-" + outputPrefix + ".model.gz"))), filteredClassifier)
}



// predictions on test
writer = new PrintWriter(new GZIPOutputStream(new FileOutputStream(new File(classifierDirectory, "test-" + outputPrefix + ".csv.gz"))))
writer.write(header)

for (instance in test) 
{
    int id = instance.value(test.attribute(idAttribute))
    double prediction
    if (!regression) 
    {
        label = (instance.stringValue(instance.classAttribute()).equals(predictClassValue)) ? 1 : 0
        prediction = filteredClassifier.distributionForInstance(instance)[predictClassIndex]
    } 
    else 
    {
        label = instance.classValue()
        prediction = filteredClassifier.distributionForInstance(instance)[0]
    }
    row = sprintf "%s,%s,%f\n", id, label, prediction
    writer.write(row)
}
writer.flush()
writer.close()



//predictions on validation
writer = new PrintWriter(new GZIPOutputStream(new FileOutputStream(new File(classifierDirectory, "valid-" + outputPrefix + ".csv.gz"))))
writer.write(header)

for (instance in validation)
{
    int id = instance.value(validation.attribute(idAttribute))
    double prediction
    if (!regression)
    {
        label = (instance.stringValue(instance.classAttribute()).equals(predictClassValue)) ? 1 : 0
        prediction = filteredClassifier.distributionForInstance(instance)[predictClassIndex]
    }
    else
    {
        label = instance.classValue()
        prediction = filteredClassifier.distributionForInstance(instance)[0]
    }
    row = sprintf "%s,%s,%f\n", id, label, prediction
    writer.write(row)
}
writer.flush()
writer.close()











