// This example shows how to save data to train the ml5 NeuralNetwork classifier to recognize
// a basic facial expression, though it can also be adapted to hand or pose gestures

let video;

let classifier;
let isTrained = false;
let classification = "";

let faceMesh;
let faceOptions = { maxFaces: 1, refineLandmarks: false, flipped: true };
let faces = [];

// Object to keep count of samples collected
let sampleCount = {};

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
    debug: true,
  };
  classifier = ml5.neuralNetwork(classifierOptions);

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
    let selectedKeypoints = [
      ...face.leftEye.keypoints,
      ...face.rightEye.keypoints,
      ...face.lips.keypoints,
      ...face.leftEyebrow.keypoints,
      ...face.rightEyebrow.keypoints,
      ...face.faceOval.keypoints
    ];
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

// Convert the model keypoints to a 1D array of just the x and y values for training
function flattenData() {
  let face = faces[0];
  let selectedKeypoints = [
    ...face.leftEye.keypoints,
    ...face.rightEye.keypoints,
    ...face.lips.keypoints,
    ...face.leftEyebrow.keypoints,
    ...face.rightEyebrow.keypoints,
    ...face.faceOval.keypoints
  ];
  let data = [];
  // Normalizing keypoints relative to the face's bounding box should improve results
  // Otherwise, training will be heavily influenced by the size and posittion of the face in the frame
  for (let keypoint of selectedKeypoints) {
    let normX = (keypoint.x - face.box.xMin) / face.box.width;
    let normY = (keypoint.y - face.box.yMin) / face.box.height;
    data.push(normX, normY);
  }
  return data;
}

//

const addButton = document.getElementById("add-button");

addButton.addEventListener("click", addData);

function addData() {
  if (faces[0]) {
    let inputData = flattenData();
    let label = document.getElementById("label-input").value;
    let outputData = [label];
    classifier.addData(inputData, outputData);

    // Keep track of the number of samples recorded with a specific label
    if (sampleCount[label]) {
      sampleCount[label]++;
    } else sampleCount[label] = 1;

    // Display sample count
    let html = "";
    for (let key of Object.keys(sampleCount)) {
      html += `<p>${key}: ${sampleCount[key]}</p>`;
      document.getElementById("sample-count-display").innerHTML = html;
    }
  }
}

const saveButton = document.getElementById("save-button");

saveButton.addEventListener("click", saveClassifier);

function saveClassifier() {
  classifier.save();
}

const trainButton = document.getElementById("train-button");
trainButton.addEventListener("click", train);

function train() {
  // The data should be normalized before training
  classifier.normalizeData();

  // Train the model
  let trainingOptions = {
    epochs: 50,
  };
  classifier.train(trainingOptions, finishedTraining);
}

function finishedTraining() {
  console.log("Training finished!");
  isTrained = true;
}
