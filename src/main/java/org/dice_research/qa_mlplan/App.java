package org.dice_research.qa_mlplan;

import java.io.File;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.learner.ISupervisedLearner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.mlplan.core.AMLPlanBuilder;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.weka.MLPlanWekaBuilder;

public class App {
	private static Logger logger = LoggerFactory.getLogger(MLPlan.class);

    public static void main( String[] args ) throws Exception {
        ArffDatasetAdapter datasetAdapter = new ArffDatasetAdapter();
        logger.info("Learn");
        ILabeledDataset trainData = datasetAdapter.readDataset(new File("data/train.arff"));
        AMLPlanBuilder builder = new MLPlanWekaBuilder().withDataset(trainData);
        MLPlan mlplan = builder.build();
        ISupervisedLearner learner = mlplan.call();
        logger.info("Learner: {}", learner);
        logger.info("Predict");
        ILabeledDataset testData = datasetAdapter.readDataset(new File("data/test.arff"));
        IPredictionBatch prediction = learner.predict(testData);
        //logger.info("Prediction: {}", prediction);
        logger.info("Done");
    }
}
