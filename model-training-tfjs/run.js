const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { performance } = require('perf_hooks');

const IMAGE_SIZE = 48;
const NUM_CLASSES = 7;
const BATCH_SIZE = 64;
const EPOCHS = 100;
const CSV_PATH = './fer2013.csv';
const TRAIN_USAGE_FILTER = 'Training';
const TEST_USAGE_FILTER = 'PublicTest';
const NUM_FEATURES = 64;

async function loadDataByUsage(filePath, usageFilter) {
    console.log(`Loading data for usage: ${usageFilter}...`);
    const rawData = fs.readFileSync(filePath, 'utf8').split('\n');
    const images = [];
    const labels = [];
    const oneHotLabels = [];
    let count = 0;

    for (let i = 1; i < rawData.length; i++) {
        const row = rawData[i].split(',');
        if (row.length < 3) continue;

        const [emotion, pixels, usage] = row.map(field => field.trim());

        if (usage !== usageFilter) continue;

        try {
            const pixelVals = pixels.split(' ').map(Number);
            if (pixelVals.length !== IMAGE_SIZE * IMAGE_SIZE) {
                continue;
            }
            const imageTensor = tf.tensor(pixelVals, [IMAGE_SIZE, IMAGE_SIZE, 1]).div(255.0);
            images.push(imageTensor);

            const emotionIndex = parseInt(emotion);
            labels.push(emotionIndex);
            oneHotLabels.push(tf.oneHot(tf.tensor1d([emotionIndex], 'int32'), NUM_CLASSES).squeeze());

            count++;
        } catch (error) {
            continue;
        }
    }

    if (images.length === 0) {
        return {
            xs: tf.tensor([], [0, IMAGE_SIZE, IMAGE_SIZE, 1]),
            ys: tf.tensor([], [0, NUM_CLASSES]),
            labels: []
        };
    }

    console.log(`Loaded ${count} samples for usage: ${usageFilter}.`);

    const xs = tf.stack(images);
    const ys = tf.stack(oneHotLabels);

    images.forEach(img => img.dispose());
    oneHotLabels.forEach(labelTensor => labelTensor.dispose());

    return { xs, ys, labels };
}

function createModel() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1],
        filters: NUM_FEATURES,
        kernelSize: 3,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l: 0.01 })
    }));
    model.add(tf.layers.conv2d({
        filters: NUM_FEATURES,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.conv2d({
        filters: 2 * NUM_FEATURES,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.conv2d({
        filters: 2 * NUM_FEATURES,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.conv2d({
        filters: 4 * NUM_FEATURES,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.conv2d({
        filters: 4 * NUM_FEATURES,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.conv2d({
        filters: 8 * NUM_FEATURES,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.conv2d({
        filters: 8 * NUM_FEATURES,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 8 * NUM_FEATURES,
        activation: 'relu'
    }));
    model.add(tf.layers.dropout({ rate: 0.4 }));
    model.add(tf.layers.dense({
        units: 4 * NUM_FEATURES,
        activation: 'relu'
    }));
    model.add(tf.layers.dropout({ rate: 0.4 }));
    model.add(tf.layers.dense({
        units: 2 * NUM_FEATURES,
        activation: 'relu'
    }));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'softmax'
    }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

async function run() {
    let trainXs, trainYs, testXs, testYs, model;
    let evalResult = null;

    try {
        const trainData = await loadDataByUsage(CSV_PATH, TRAIN_USAGE_FILTER);
        trainXs = trainData.xs;
        trainYs = trainData.ys;

        const testData = await loadDataByUsage(CSV_PATH, TEST_USAGE_FILTER);
        testXs = testData.xs;
        testYs = testData.ys;

        if (trainXs.shape[0] === 0) {
            throw new Error("No training data available. Cannot train the model.");
        }

        console.log('\nCreating model...');
        model = createModel();
        model.summary();

        console.log('\nTraining model...');
        const startTime = performance.now();

        await model.fit(trainXs, trainYs, {
            batchSize: BATCH_SIZE,
            epochs: EPOCHS,
            validationSplit: 0.1,
            shuffle: true,
            callbacks: [
                new tf.EarlyStopping({ monitor: 'val_loss', patience: 10, verbose: 1 })
            ]
        });

        const endTime = performance.now();
        const trainingTime = (endTime - startTime) / 1000;
        console.log(`Training finished in ${trainingTime.toFixed(2)} seconds.`);

        console.log('\nEvaluating model on test data...');
        if (testXs.shape[0] > 0) {
            evalResult = model.evaluate(testXs, testYs, { batchSize: BATCH_SIZE });

            const testLoss = evalResult[0].dataSync()[0];
            const testAccuracy = evalResult[1].dataSync()[0];

            console.log(`Test Loss: ${testLoss.toFixed(4)}`);
            console.log(`Test Accuracy: ${testAccuracy.toFixed(4)}`);
        } else {
            console.warn('No test data available for evaluation.');
        }

        console.log('\nSaving model in browser-compatible format...');
        await model.save('file://./face_emotion_model_browser');
        console.log('Model saved to ./face_emotion_model_browser');

    } catch (err) {
        console.error("Error during execution:", err);
    } finally {
        console.log('\nDisposing tensors and model...');
        if (trainXs) trainXs.dispose();
        if (trainYs) trainYs.dispose();
        if (testXs) testXs.dispose();
        if (testYs) testYs.dispose();
        if (evalResult) {
            evalResult.forEach(t => t.dispose());
        }
        if (model) model.dispose();
        console.log('TensorFlow.js backend memory usage:', tf.memory());
    }
}

run();