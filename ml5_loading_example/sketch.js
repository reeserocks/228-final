let video;

let classifier;
let isTrained = false;
let classification = "";

let faceMesh;
let faceOptions = { maxFaces: 1, refineLandmarks: false, flipped: true };
let faces = [];

async function setup() {
  // Load the model
  faceMesh = await ml5.faceMesh(faceOptions);

  createCanvas(640, 480);
  textSize(24);

  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  // Set up the neural network
  ml5.setBackend("webgl");
  let classifierOptions = {
    task: "classification",
  };
  classifier = ml5.neuralNetwork(classifierOptions);

  // Define the paths to the 3 downloaded files
  const modelDetails = {
    model: "model/model.json",
    metadata: "model/model_meta.json",
    weights: "model/model.weights.bin",
  };

  // Load the model and mark it ready
  await classifier.load(modelDetails);
  isTrained = true;

  // Start detection
  faceMesh.detectStart(video, (results) => (faces = results));
}

function draw() {
  push();
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0);
  pop();

  // Draw the keypoints
  // Since the faceMesh includes hundreds of keypoints, we're going to limit our focus
  // to just the eyes and lips for basic facial expression training
  // This is not necessary for the smaller hand and pose models
  if (faces[0]) {
    let face = faces[0];
    let selectedKeypoints = [...face.leftEye.keypoints, ...face.rightEye.keypoints, ...face.lips.keypoints]
    for (let keypoint of selectedKeypoints) {
      fill(0, 255, 0);
      noStroke();
      circle(keypoint.x, keypoint.y, 5);
    }
  }

  // If the model is trained, make a classification and display the results
  if (isTrained && faces[0]) {
    let inputData = flattenData();
    classifier.classify(inputData, gotClassification);

    textSize(64);
    fill(0, 255, 0);
    text(classification, 20, 60);
  }
}

// Callback function for when the classifier makes a classification
function gotClassification(results) {
  // results[0] will always be the label with the highest confidence
  // You may optionally create a confidence threshold
  classification = `${results[0].label}: ${results[0].confidence}`;
}

// Keep the EXACT same flattenData() function used during training!
function flattenData() {
  let face = faces[0];
  let selectedKeypoints = [
    ...face.leftEye.keypoints,
    ...face.rightEye.keypoints,
    ...face.lips.keypoints,
  ];
  let data = [];
  for (let keypoint of selectedKeypoints) {
    let normX = (keypoint.x - face.box.xMin) / face.box.width;
    let normY = (keypoint.y - face.box.yMin) / face.box.height;
    data.push(normX, normY);
  }
  return data;
}
