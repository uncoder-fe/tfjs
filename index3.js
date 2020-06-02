async function app() {
  console.log("Loading mobilenet..");

  // Load the model.
  //  识别
  const net = await mobilenet.load();
  //   分类
  const classifier = await knnClassifier.create();
  console.log("Successfully loaded model");

  // Make a prediction through the model on our image.
  //   const imgEl = document.getElementById("img");
  //   const result = await net.classify(imgEl);
  //   console.log(result);

  const webcamElement = document.getElementById("webcam");
  const webcam = await tf.data.webcam(webcamElement);

  const addExample = async (classId) => {
    // Capture an image from the web camera.
    const img = await webcam.capture();
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(img, true);
    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
    // Dispose the tensor to release the memory.
    img.dispose();
  };
  const uploadModel = async (classifierModel, event) => {
    const inputModel = event.target.files;
    console.log("上传中");
    const fr = new FileReader();
    if (inputModel.length > 0) {
      fr.onload = async () => {
        var dataset = fr.result;
        var tensorObj = JSON.parse(dataset);
        Object.keys(tensorObj).forEach((key) => {
          tensorObj[key] = tf.tensor(tensorObj[key], [
            tensorObj[key].length / 1024,
            1024,
          ]);
        });
        classifierModel.setClassifierDataset(tensorObj);
        console.log("分类器设置成功! ");
      };
    }
    await fr.readAsText(inputModel[0]);
    console.log("上传完成");
  };
  const downloadModel = async (classifierModel) => {
    let datasets = await classifierModel.getClassifierDataset();
    let datasetObject = {};
    Object.keys(datasets).forEach((key) => {
      let data = datasets[key].dataSync();
      datasetObject[key] = Array.from(data);
    });
    let jsonModel = JSON.stringify(datasetObject);
    let downloader = document.createElement("a");
    downloader.download = "model.json";
    downloader.href =
      "data:text/text;charset=utf-8," + encodeURIComponent(jsonModel);
    document.body.appendChild(downloader);
    downloader.click();
    downloader.remove();
  };
  // When clicking a button, add an example for that class.
  document
    .getElementById("class-a")
    .addEventListener("click", () => addExample(0));
  document
    .getElementById("class-b")
    .addEventListener("click", () => addExample(1));
  document
    .getElementById("class-c")
    .addEventListener("click", () => addExample(2));
  document
    .getElementById("load_button")
    .addEventListener("change", (event) => uploadModel(classifier, event));
  document
    .getElementById("save_button")
    .addEventListener("click", async () => downloadModel(classifier));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, "conv_preds");
      // Get the most likely class and confidence from the classifier module.
      const result = await classifier.predictClass(activation);
      const classes = ["人", "手", "口罩包装"];
      document.getElementById("console").innerText = `
            prediction: ${classes[result.label]}\n
            probability: ${result.confidences[result.label]}
          `;
      // Dispose the tensor to release the memory.
      img.dispose();
    }

    await tf.nextFrame();
  }
}

app();
