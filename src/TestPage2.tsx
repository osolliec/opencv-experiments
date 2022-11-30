import React from "react";
// Adds the CPU backend.
import cv from "@techstark/opencv-js";

// Import @tensorflow/tfjs-core
import * as tf from "@tensorflow/tfjs-core";
// Import @tensorflow/tfjs-tflite.
import * as tflite from "@tensorflow/tfjs-tflite";
import "@tensorflow/tfjs-backend-cpu";

// @ts-ignore

class TestPage2 extends React.Component {
  model: null | tflite.TFLiteModel;
  // @ts-ignore
  constructor(props) {
    super(props);
    this.state = {
      imgUrl: null,
    };
    this.model = null;
  }

  async loadModel() {
    const tfliteModel = await tflite.loadTFLiteModel("public/puffy-laugh-metadata.tflite");
    console.log(tfliteModel);
    return tfliteModel;
  }

  async componentDidMount() {
    //loadHaarFaceModels();
    this.model = await this.loadModel();
  }

  predict(video: HTMLCanvasElement | null) {
    if (!video) return;
    if (!this.model) {
      console.log("no model");
      return;
    }
    // Prepare input tensors.
    const img = tf.browser.fromPixels(video);
    console.log(img);
    const expandDim = tf.expandDims(img);
    console.log(expandDim);
    const div = tf.div(expandDim, 127.5);
    console.log(div);
    let input = tf.sub(div, 1);
    console.log(input);

    //@ts-ignore
    input = tf.cast(input, "bool");
    console.log(input);

    // Run inference and get output tensors.
    let outputTensor = this.model.predict(input) as tf.Tensor;
    console.log("here");
    console.log(outputTensor.dataSync());
  }

  handleCamera() {
    var vgaConstraints = {
      video: {
        width: 224,
        height: 224,
      },
    };
    // @ts-ignore
    let video: HTMLVideoElement | null = document.getElementById("videoInput"); // video is the id of video tag
    // @ts-ignore

    let canvas: HTMLCanvasElement | null = document.getElementById("canvasOutput");

    const that = this;
    navigator.mediaDevices
      .getUserMedia(vgaConstraints)
      .then(function (stream) {
        if (video === null) {
          return;
        }
        video.srcObject = stream;
        video.play();

        const height = 224;
        const width = 224;

        const size = new cv.Size(width, height);
        const mat_type = cv.CV_8UC4;

        let original = new cv.Mat(size, mat_type);
        //let gray = new cv.Mat(size, mat_type);

        let cap = new cv.VideoCapture(video);

        const FPS = 24;

        function processVideo() {
          let begin = Date.now();

          cap.read(original);
          // cv.cvtColor(original, gray, cv.COLOR_RGBA2GRAY);
          // cv.imshow("canvasOutput", gray);

          // @ts-ignore
          that.predict(video);

          let delay = 1000 / FPS - (Date.now() - begin);
          setTimeout(processVideo, delay);
        }
        // schedule first one.
        setTimeout(processVideo, 2000);
      })
      .catch(function (err) {
        console.error(err);
      });
  }

  render() {
    return (
      <div>
        <div style={{ marginTop: "30px" }}>
          <button onClick={() => this.handleCamera()}>Click me to start camera</button>
        </div>
      </div>
    );
  }
}

export default TestPage2;
