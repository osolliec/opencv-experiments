import React from "react";
import cv from "@techstark/opencv-js";
import { loadHaarFaceModels, detectHaarFace } from "./haarFaceDetection";
import { withEdge } from "./withEdge";

// @ts-ignore
window.cv = cv;
const color = [255, 0, 0, 255];

const params = {
  buffer_length: 50,
  line_weight: 10,
  line_check_region: 20,
  space_num: 7,
  maxLineGap: 10,
  minLineLength: 200,
  hough_line_treshold: 40,
  hough_line_theta: Math.PI / 180,
  angle_tolerance: 10,
  // init box
  top_left_x: 200,
  top_left_y: 150,
  top_right_x: 510,
  top_right_y: 150,
  bottom_right_x: 510,
  bottom_right_y: 340,
  bottom_left_x: 200,
  bottom_left_y: 340,
};

type Line = Array<Number>;
type LineBuffer = {
  leftLines: Array<Line>;
  topLines: Array<Line>;
  rightLines: Array<Line>;
  bottomLines: Array<Line>;
};

const drawRectangle = (src: cv.Mat) => {
  cv.line(
    src,
    new cv.Point(params.top_left_x, params.top_left_y),
    new cv.Point(params.top_right_x, params.top_right_y),
    color,
    params.line_weight
  );

  cv.line(
    src,
    new cv.Point(params.top_right_x, params.top_right_y),
    new cv.Point(params.bottom_right_x, params.bottom_right_y),
    color,
    params.line_weight
  );
  cv.line(
    src,
    new cv.Point(params.bottom_right_x, params.bottom_right_y),
    new cv.Point(params.bottom_left_x, params.bottom_left_y),
    color,
    params.line_weight
  );
  cv.line(
    src,
    new cv.Point(params.bottom_left_x, params.bottom_left_y),
    new cv.Point(params.top_left_x, params.top_left_y),
    color,
    params.line_weight
  );

  return src;
};

const doCanny = (src: cv.Mat, dst: cv.Mat) => {
  cv.Canny(src, dst, 10, 30, 3);
};

const doHoughLines = (
  src: cv.Mat,
  dst: cv.Mat,
  listCount: number,
  lineBuffer: LineBuffer
) => {
  const lines = new cv.Mat();

  cv.HoughLinesP(
    src,
    lines,
    1,
    params.hough_line_theta,
    params.hough_line_treshold,
    params.minLineLength,
    params.maxLineGap
    // params.hough_line_theta,
    // params.hough_line_treshold,
    // params.minLineLength,
    // params.maxLineGap
  );

  // TODO need to clear the lines

  console.log(lines.size());
  console.log(lines.rows);

  // collecting lines algorithm
  // for eachline in lines
  for (let i = 0; i < lines.rows; ++i) {
    // for x1, y1, x2, y2 in eachline
    let startPoint = new cv.Point(
      lines.data32S[i * 4],
      lines.data32S[i * 4 + 1]
    );

    let endPoint = new cv.Point(
      lines.data32S[i * 4 + 2],
      lines.data32S[i * 4 + 3]
    );

    const x1 = startPoint.x;
    const x2 = endPoint.x;
    const y1 = startPoint.y;
    const y2 = endPoint.y;

    console.log("here");

    if (y1 - y2 !== 0) {
      // collecting left line
      if (
        Math.abs(x1 - params.top_left_x) < params.line_check_region &&
        params.top_left_y < y1 &&
        y1 < params.bottom_left_y &&
        params.top_left_y < y2 &&
        y2 < params.bottom_left_y
      ) {
        const angle1 = Math.atan2(y2 - y1, x2 - x1);
        const angle1Degrees = (angle1 * 360) / (2 * Math.PI);

        if (
          (90 - params.angle_tolerance < angle1Degrees &&
            angle1Degrees < 90 + params.angle_tolerance) ||
          (-90 - params.angle_tolerance < angle1Degrees &&
            angle1Degrees < -90 + params.angle_tolerance)
        ) {
          lineBuffer.leftLines.push([listCount, x1, y1, x2, y2]);
        }
      }

      // collecting right line
      if (
        Math.abs(x1 - params.top_right_x) < params.line_check_region &&
        params.top_left_y < y1 &&
        y1 < params.bottom_left_y &&
        params.top_left_y < y2 &&
        y2 < params.bottom_left_y
      ) {
        const angle1 = Math.atan2(y2 - y1, x2 - x1);
        const angle1Degrees = (angle1 * 360) / (2 * Math.PI);
        if (
          (90 - params.angle_tolerance < angle1Degrees &&
            angle1Degrees < 90 + params.angle_tolerance) ||
          (-90 - params.angle_tolerance < angle1Degrees &&
            angle1Degrees < -90 + params.angle_tolerance)
        ) {
          lineBuffer.rightLines.push([listCount, x1, y1, x2, y2]);
        }
      }
    }

    if (x1 - x2 !== 0) {
      //# collecting top line
      if (
        Math.abs(y1 - params.top_left_y) < params.line_check_region &&
        params.top_left_x < x1 &&
        x1 < params.top_right_x &&
        params.top_left_x < x2 &&
        x2 < params.top_right_x
      ) {
        const angle1 = Math.atan2(y2 - y1, x2 - x1);
        const angle1Degrees = (angle1 * 360) / (2 * Math.PI);
        if (
          0 - params.angle_tolerance < angle1Degrees &&
          angle1Degrees < 0 + params.angle_tolerance
        ) {
          lineBuffer.topLines.push([listCount, x1, y1, x2, y2]);
        }
      }

      //# collecting bottom line
      if (
        Math.abs(y1 - params.bottom_left_y) < params.line_check_region &&
        params.top_left_x < x1 &&
        x1 < params.top_right_x &&
        params.top_left_x < x2 &&
        x2 < params.top_right_x
      ) {
        const angle1 = Math.atan2(y2 - y1, x2 - x1);
        const angle1Degrees = (angle1 * 360) / (2 * Math.PI);
        if (
          0 - params.angle_tolerance < angle1Degrees &&
          angle1Degrees < 0 + params.angle_tolerance
        ) {
          lineBuffer.bottomLines.push([listCount, x1, y1, x2, y2]);
        }
      }
    }

    cv.line(dst, startPoint, endPoint, color, params.line_weight);
  }
};

class TestPage extends React.Component {
  // @ts-ignore
  constructor(props) {
    super(props);
    this.state = {
      imgUrl: null,
    };
  }

  componentDidMount() {
    //loadHaarFaceModels();
  }

  /////////////////////////////////////////
  //
  // process image with opencv.js
  //
  /////////////////////////////////////////
  // processImage(imgSrc) {
  //   const img = cv.imread(imgSrc);

  //   // to gray scale
  //   const imgGray = new cv.Mat();
  //   cv.cvtColor(img, imgGray, cv.COLOR_BGR2GRAY);
  //   cv.imshow(this.grayImgRef.current, imgGray);

  //   // detect edges using Canny
  //   const edges = new cv.Mat();
  //   cv.Canny(imgGray, edges, 100, 100);
  //   cv.imshow(this.cannyEdgeRef.current, edges);

  //   // test
  //   const detectEdges = withEdge(imgSrc);
  //   cv.imshow(this.edgeRef.current, detectEdges);

  //   // detect faces using Haar-cascade Detection
  //   // const haarFaces = detectHaarFace(img);
  //   // cv.imshow(this.haarFaceImgRef.current, haarFaces);

  //   // need to release them manually
  //   img.delete();
  //   imgGray.delete();
  //   edges.delete();
  //   //haarFaces.delete();
  // }

  handleCamera() {
    // @ts-ignore
    let video: HTMLVideoElement | null = document.getElementById("videoInput"); // video is the id of video tag

    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then(function (stream) {
        console.log(stream);
        if (video === null) {
          return;
        }
        video.srcObject = stream;
        video.play();

        const height = 480;
        const width = 640;

        const size = new cv.Size(width, height);
        const mat_type = cv.CV_8UC4;

        let original = new cv.Mat(size, mat_type);
        let gray = new cv.Mat(size, mat_type);
        let canny = new cv.Mat(size, mat_type);
        let edge = new cv.Mat(size, mat_type);

        let cap = new cv.VideoCapture(video);

        const FPS = 24;

        const linesList = {
          leftLines: [],
          topLines: [],
          rightLines: [],
          bottomLines: [],
        };

        const listCount = 0;

        function processVideo() {
          let begin = Date.now();
          cap.read(original);
          cv.cvtColor(original, gray, cv.COLOR_RGBA2GRAY);

          //const withRect = drawRectangle(original.clone());
          //cv.imshow("canvasOutput2", withRect);

          doCanny(gray, canny);
          cv.imshow("canvasOutput", canny);

          doHoughLines(canny, original, listCount, linesList);
          cv.imshow("canvasOutput2", original);
          //console.log(JSON.stringify(lines.data));
          //cv.imshow("canvasOutput4", lines);

          // schedule next one.
          let delay = 1000 / FPS - (Date.now() - begin);
          setTimeout(processVideo, delay);
        }
        // schedule first one.
        setTimeout(processVideo, 0);
      })
      .catch(function (err) {
        console.error(err);
        //console.log("An error occurred! " + err);
      });
  }

  render() {
    return (
      <div>
        <div style={{ marginTop: "30px" }}>
          <button onClick={() => this.handleCamera()}>
            Click me to start camera
          </button>
        </div>
      </div>
    );
  }
}

export default TestPage;
