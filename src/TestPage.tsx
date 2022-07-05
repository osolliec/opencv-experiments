import React from "react";
import cv from "@techstark/opencv-js";
import { loadHaarFaceModels, detectHaarFace } from "./haarFaceDetection";
import { withEdge } from "./withEdge";

// @ts-ignore
window.cv = cv;
const blue = [0, 0, 255, 255];
const red = [255, 0, 0, 255];

const params = {
  buffer_length: 50,
  line_weight: 10,
  line_check_region: 30,
  space_num: 7,
  maxLineGap: 3,
  minLineLength: 30,
  hough_line_treshold: 150,
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

type Line = Array<number>;
type LineBuffer = {
  leftLines: Array<Line>;
  topLines: Array<Line>;
  rightLines: Array<Line>;
  bottomLines: Array<Line>;
};

const drawRectangle = (dst: cv.Mat) => {
  cv.line(
    dst,
    new cv.Point(params.top_left_x, params.top_left_y),
    new cv.Point(params.top_right_x, params.top_right_y),
    red,
    params.line_weight
  );

  cv.line(
    dst,
    new cv.Point(params.top_right_x, params.top_right_y),
    new cv.Point(params.bottom_right_x, params.bottom_right_y),
    red,
    params.line_weight
  );
  cv.line(
    dst,
    new cv.Point(params.bottom_right_x, params.bottom_right_y),
    new cv.Point(params.bottom_left_x, params.bottom_left_y),
    red,
    params.line_weight
  );
  cv.line(
    dst,
    new cv.Point(params.bottom_left_x, params.bottom_left_y),
    new cv.Point(params.top_left_x, params.top_left_y),
    red,
    params.line_weight
  );
};

const doCanny = (src: cv.Mat, dst: cv.Mat) => {
  cv.Canny(src, dst, 10, 30, 3);
};

// clear the lines
const clearLines = (iterationCount: number, lines: Array<Line>) => {
  let i = lines.length;
  if (i === 0) return;

  // clear from end of array because we remove elements from the array
  while (i--) {
    if (iterationCount - lines[i][0] >= params.buffer_length + 1) {
      const removed = lines.splice(i, 1);
    }
  }
};

const doHoughLines = (src: cv.Mat, dst: cv.Mat, iterationCount: number, lineBuffer: LineBuffer) => {
  const lines = new cv.Mat();

  cv.HoughLinesP(
    src,
    lines,
    1,
    params.hough_line_theta,
    params.hough_line_treshold,
    params.minLineLength,
    params.maxLineGap
  );

  // collecting lines algorithm
  // for eachline in lines
  for (let i = 0; i < lines.rows; ++i) {
    // for x1, y1, x2, y2 in eachline
    let startPoint = new cv.Point(lines.data32S[i * 4], lines.data32S[i * 4 + 1]);

    let endPoint = new cv.Point(lines.data32S[i * 4 + 2], lines.data32S[i * 4 + 3]);

    const x1 = startPoint.x;
    const x2 = endPoint.x;
    const y1 = startPoint.y;
    const y2 = endPoint.y;

    if (y1 - y2 !== 0) {
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
          (90 - params.angle_tolerance < angle1Degrees && angle1Degrees < 90 + params.angle_tolerance) ||
          (-90 - params.angle_tolerance < angle1Degrees && angle1Degrees < -90 + params.angle_tolerance)
        ) {
          lineBuffer.leftLines.push([iterationCount, x1, y1, x2, y2]);
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
          (90 - params.angle_tolerance < angle1Degrees && angle1Degrees < 90 + params.angle_tolerance) ||
          (-90 - params.angle_tolerance < angle1Degrees && angle1Degrees < -90 + params.angle_tolerance)
        ) {
          lineBuffer.rightLines.push([iterationCount, x1, y1, x2, y2]);
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
        if (0 - params.angle_tolerance < angle1Degrees && angle1Degrees < 0 + params.angle_tolerance) {
          lineBuffer.topLines.push([iterationCount, x1, y1, x2, y2]);
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
        if (0 - params.angle_tolerance < angle1Degrees && angle1Degrees < 0 + params.angle_tolerance) {
          lineBuffer.bottomLines.push([iterationCount, x1, y1, x2, y2]);
        }
      }
    }

    cv.line(dst, startPoint, endPoint, red, params.line_weight);
  }
};

const linspace = (startValue: number, stopValue: number, cardinality: number) => {
  const arr: Array<number> = [];
  const step = (stopValue - startValue) / (cardinality - 1);
  for (var i = 0; i < cardinality; i++) {
    arr.push(startValue + step * i);
  }
  return arr;
};

const zeros = (nb: number) => {
  const arr: Array<number> = [];
  for (var i = 0; i < nb; i++) {
    arr.push(0);
  }
  return arr;
};

// returns all indices where cond is true for the val of the array
const where = (array: Array<number>, condition: (number: number) => boolean): Array<number> => {
  const arr: Array<number> = [];
  array.forEach((val, index) => {
    if (condition(val)) {
      arr.push(index);
    }
  });

  return arr;
};

const setArray = (
  dstArray: Array<number>,
  indexArray: Array<number>,
  condition: (number: number) => boolean,
  val: number
) => {
  let indexToSet = where(indexArray, condition);

  indexToSet.forEach((index) => {
    dstArray[index] = val;
  });
};

const setFirstOfArray = (
  dstArray: Array<number>,
  indexArray: Array<number>,
  condition: (number: number) => boolean,
  val: number
) => {
  let indexToSet = where(indexArray, condition);
  dstArray[indexToSet[0]] = val;
};

const sum = (array: Array<number>) => {
  return array.reduce((curSum, a) => curSum + a, 0);
};

const round = (number: number) => {
  return Math.round(number * 100) / 100;
};

const visualizeLines = (dst: cv.Mat, lineBuffer: LineBuffer) => {
  const left_line_space_index = linspace(params.top_left_y, params.bottom_left_y, params.space_num);
  const left_line_space_array = zeros(left_line_space_index.length - 1);

  lineBuffer.leftLines.forEach((line) => {
    const x1 = params.top_left_x;
    const y1 = line[2];
    const x2 = params.top_left_x;
    const y2 = line[4];

    if (y1 <= y2) {
      // left_line_space_array[np.where((left_line_space_index > y1) & (left_line_space_index < y2))] = 1
      setArray(left_line_space_array, left_line_space_index, (n) => n > y1 && n < y2, 1);
      // left_line_space_array[np.where((left_line_space_index < y1))[0][0]] = 1
      setFirstOfArray(left_line_space_array, left_line_space_index, (n) => n < y1, 1);
    } else {
      // left_line_space_array[np.where((left_line_space_index < y1) & (left_line_space_index > y2))] = 1
      setArray(left_line_space_array, left_line_space_index, (n) => n < y1 && n > y2, 1);
      // left_line_space_array[np.where((left_line_space_index < y2))[0][0]] = 1
      setFirstOfArray(left_line_space_array, left_line_space_index, (n) => n < y2, 1);
    }
  });

  const top_line_space_index = linspace(params.top_left_x, params.top_right_x, params.space_num);

  const top_line_space_array = zeros(left_line_space_index.length - 1);

  lineBuffer.topLines.forEach((line) => {
    const x1 = line[1];
    const y1 = params.top_left_y;
    const x2 = line[3];
    const y2 = params.top_left_y;

    if (x1 <= x2) {
      // top_line_space_array[np.where((top_line_space_index < x2) & (top_line_space_index > x1))] = 1
      setArray(top_line_space_array, top_line_space_index, (n) => n < x2 && n > x1, 1);
      // top_line_space_array[np.where((top_line_space_index < x1))[0][0]] = 1
      setFirstOfArray(top_line_space_array, top_line_space_index, (n) => n < x1, 1);
    } else {
      // top_line_space_array[np.where((top_line_space_index < x1) & (top_line_space_index > x2))] = 1
      setArray(top_line_space_array, top_line_space_index, (n) => n < x1 && n > x2, 1);
      // top_line_space_array[np.where((top_line_space_index < x2))[0][0]] = 1
      setFirstOfArray(top_line_space_array, top_line_space_index, (n) => n < x2, 1);
    }
  });

  const right_line_space_index = linspace(params.top_right_y, params.bottom_right_y, params.space_num);
  const right_line_space_array = zeros(right_line_space_index.length - 1);
  lineBuffer.rightLines.forEach((line) => {
    const x1 = params.top_right_x;
    const y1 = line[2];
    const x2 = params.top_right_x;
    const y2 = line[4];

    if (y1 <= y2) {
      // right_line_space_array[np.where((right_line_space_index > y1) & (right_line_space_index < y2))] = 1;
      setArray(right_line_space_array, right_line_space_index, (n) => n > y1 && n < y2, 1);
      // right_line_space_array[np.where(right_line_space_index < y1)[0][0]] = 1;
      setFirstOfArray(right_line_space_array, right_line_space_index, (n) => n < y1, 1);
    } else {
      // right_line_space_array[np.where((right_line_space_index < y1) & (right_line_space_index > y2))] = 1;
      setArray(right_line_space_array, right_line_space_index, (n) => n < y1 && n > y2, 1);
      // right_line_space_array[np.where(right_line_space_index < y2)[0][0]] = 1;
      setFirstOfArray(right_line_space_array, right_line_space_index, (n) => n < y2, 1);
    }
  });

  const bottom_line_space_index = linspace(params.bottom_left_x, params.bottom_right_x, params.space_num);
  const bottom_line_space_array = zeros(bottom_line_space_index.length - 1);

  lineBuffer.bottomLines.forEach((line) => {
    const x1 = line[1];
    const y1 = params.bottom_right_y;
    const x2 = line[3];
    const y2 = params.bottom_right_y;
    if (x1 <= x2) {
      // bottom_line_space_array[np.where((bottom_line_space_index < x2) & (bottom_line_space_index > x1))] = 1
      setArray(bottom_line_space_array, bottom_line_space_index, (n) => n < x2 && n > x1, 1);
      // bottom_line_space_array[np.where((bottom_line_space_index < x1))[0][0]] = 1
      setFirstOfArray(bottom_line_space_array, bottom_line_space_index, (n) => n < x1, 1);
    } else {
      // bottom_line_space_array[np.where((bottom_line_space_index > x2) & (bottom_line_space_index < x1))] = 1
      setArray(bottom_line_space_array, bottom_line_space_index, (n) => n > x2 && n < x1, 1);
      // bottom_line_space_array[np.where((bottom_line_space_index < x2))[0][0]] = 1
      setFirstOfArray(bottom_line_space_array, bottom_line_space_index, (n) => n < x2, 1);
    }
  });

  // visualize line segments
  left_line_space_array.forEach((val, index) => {
    if (val === 1) {
      const x1 = params.top_left_x;
      const y1 = Math.floor(left_line_space_index[index]);
      const x2 = params.top_left_x;
      const y2 = Math.floor(left_line_space_index[index + 1]);

      cv.line(dst, new cv.Point(x1, y1), new cv.Point(x2, y2), blue, params.line_weight);
    }
  });

  top_line_space_array.forEach((val, index) => {
    if (val === 1) {
      const x1 = Math.floor(top_line_space_index[index]);
      const y1 = params.top_left_y;
      const x2 = Math.floor(top_line_space_index[index + 1]);
      const y2 = params.top_left_y;

      cv.line(dst, new cv.Point(x1, y1), new cv.Point(x2, y2), blue, params.line_weight);
    }
  });

  right_line_space_array.forEach((val, index) => {
    console.log("index " + index);
    console.log("val " + val);
    if (val === 1) {
      const x1 = params.top_right_x;
      const y1 = Math.floor(right_line_space_index[index]);
      const x2 = params.top_right_x;
      const y2 = Math.floor(right_line_space_index[index + 1]);

      cv.line(dst, new cv.Point(x1, y1), new cv.Point(x2, y2), blue, params.line_weight);
    }
  });

  bottom_line_space_array.forEach((val, index) => {
    if (val === 1) {
      const x1 = Math.floor(bottom_line_space_index[index]);
      const y1 = params.bottom_right_y;
      const x2 = Math.floor(bottom_line_space_index[index + 1]);
      const y2 = params.bottom_right_y;

      cv.line(dst, new cv.Point(x1, y1), new cv.Point(x2, y2), blue, params.line_weight);
    }
  });

  // visualize score
  const left_score = round(sum(left_line_space_array) / left_line_space_array.length);
  cv.putText(dst, left_score + "", new cv.Point(100, 250), cv.FONT_HERSHEY_SIMPLEX, 1, red, params.line_weight - 5);
  const top_score = round(sum(top_line_space_array) / top_line_space_array.length);
  cv.putText(dst, top_score + "", new cv.Point(300, 100), cv.FONT_HERSHEY_SIMPLEX, 1, red, params.line_weight - 5);
  const right_score = round(sum(right_line_space_array) / top_line_space_array.length);
  cv.putText(dst, right_score + "", new cv.Point(550, 250), cv.FONT_HERSHEY_SIMPLEX, 1, red, params.line_weight - 5);
  const bottom_score = round(sum(bottom_line_space_array) / top_line_space_array.length);
  cv.putText(dst, bottom_score + "", new cv.Point(300, 430), cv.FONT_HERSHEY_SIMPLEX, 1, red, params.line_weight - 5);

  const total_score = left_score + top_score + right_score + bottom_score;
  cv.putText(dst, total_score + "", new cv.Point(300, 250), cv.FONT_HERSHEY_SIMPLEX, 1, red, params.line_weight - 5);
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
        //let frame_lines = new cv.Mat(size, mat_type);
        //let boxed_frames = new cv.Mat(size, mat_type);
        let frame_canny = new cv.Mat(size, mat_type);

        let cap = new cv.VideoCapture(video);

        const FPS = 24;

        const lineBuffer = {
          leftLines: [],
          topLines: [],
          rightLines: [],
          bottomLines: [],
        };

        let iterationCount = -1;

        function processVideo() {
          iterationCount++;
          let begin = Date.now();
          cap.read(original);
          cv.cvtColor(original, gray, cv.COLOR_RGBA2GRAY);

          const boxed_frames = original.clone();
          drawRectangle(boxed_frames);
          //cv.imshow("canvasOutput3", boxed_frames);

          doCanny(gray, frame_canny);
          cv.imshow("canvasOutput", frame_canny);

          clearLines(iterationCount, lineBuffer.bottomLines);
          clearLines(iterationCount, lineBuffer.topLines);
          clearLines(iterationCount, lineBuffer.leftLines);
          clearLines(iterationCount, lineBuffer.rightLines);

          const frame_lines = original.clone();
          doHoughLines(frame_canny, frame_lines, iterationCount, lineBuffer);
          cv.imshow("canvasOutput2", frame_lines);

          visualizeLines(boxed_frames, lineBuffer);
          cv.imshow("canvasOutput3", boxed_frames);

          // schedule next one.
          let delay = 1000 / FPS - (Date.now() - begin);
          setTimeout(processVideo, delay);
        }
        // schedule first one.
        setTimeout(processVideo, 0);
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

export default TestPage;
