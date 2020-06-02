// import * as tf from "@tensorflow/tfjs";
import { MnistData } from "./data.js";
const clearBtn = document.querySelector("#clear");
const canvas = document.querySelector("#myCanvas");
const ctx = canvas.getContext("2d");
ctx.fillRect(0, 0, 28, 28);

async function showExamples(data) {
  // Create a container in the visor
  const surface = tfvis
    .visor()
    .surface({ name: "输入数据样本", tab: "Input Data" });

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // 重塑 the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = "margin: 4px;";
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}
function getModel() {
  const model = tf.sequential();

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // 在我们的卷积神经网络的第一层中，我们有以下内容来指定输入形状。然后我们指定一些参数，用于在此层进行的卷积运算。
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    })
  );

  // MaxPooling层作为一种下采样，使用区域内的最大值来代替平均化，而不是平均化。
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  //重复另一个 conv2d + maxPooling 堆栈。注意，我们在卷积里有更多的滤波器。
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // 现在我们将二维滤波器的输出平铺成一维向量，准备输入到最后一层。这是将高维数据输入到最后的分类输出层时的常见做法。
  model.add(tf.layers.flatten());

  //我们的最后一层是一个密集层，它有10个输出单元，每个输出类有一个输出单元。(i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: "varianceScaling",
      activation: "softmax",
    })
  );

  // 选择一个优化器、损失函数和精度指标，然后编译并返回模型。
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}
async function train(model, data) {
  const metrics = ["loss", "val_loss", "acc", "val_acc"];
  const container = {
    name: "模型训练",
    styles: { height: "1000px" },
  };

  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks,
  });
}
function drawLine(data = []) {
  // console.log("data", data);
  ctx.clearRect(0, 0, 28, 28);
  ctx.beginPath();
  ctx.fillRect(0, 0, 28, 28);
  for (let i = 0; i < data.length; i++) {
    ctx.moveTo(data[i][0], data[i][1]);
    if (data[i + 1]) {
      ctx.lineTo(data[i + 1][0], data[i + 1][1]);
    }
  }
  ctx.strokeStyle = "#fff";
  ctx.stroke();
}
async function run() {
  const data = new MnistData();
  await data.load();
  await showExamples(data);
  const model = getModel();
  tfvis.show.modelSummary({ name: "模型结构" }, model);
  await train(model, data);
  console.log("训练结束");
  const points = [];
  let isClick = false;
  canvas.addEventListener("mousedown", (event) => {
    points.length = 0;
    console.log(points);
    points.push([event.offsetX, event.offsetY]);
    isClick = true;
  });
  canvas.addEventListener("mousemove", () => {
    if (isClick) {
      points.push([event.offsetX, event.offsetY]);
      drawLine(points);
    }
  });
  canvas.addEventListener("mouseup", () => {
    isClick = false;
    const data = ctx.getImageData(0, 0, 28, 28).data;
    const input = [];
    for (let i = 0; i < data.length; i += 4) {
      input.push(data[i] / 255);
    }
    model
      .predict([tf.tensor(input).reshape([1, 28, 28, 1])])
      .array()
      .then(function (scores) {
        const predicted = scores[0].indexOf(Math.max(...scores[0]));
        console.log(scores[0], predicted);
        document.querySelector("#number").innerHTML = `可能是数字:${predicted}`;
      });
  });
  clearBtn.addEventListener("click", () => {
    ctx.clearRect(0, 0, 28, 28);
  });
}

document.addEventListener("DOMContentLoaded", run);
