import cv, { Matx_MulOp } from "@techstark/opencv-js";
// see https://github.com/adityaguptai/Document-Boundary-Detection/blob/master/scan.py

type SuperContour = {
  contour: cv.Mat;
  area: number;
};
const withEdge = (imgSrc: string) => {
  console.log(imgSrc);
  const img = cv.imread(imgSrc);
  const imgGray = new cv.Mat();

  cv.cvtColor(img, imgGray, cv.COLOR_BGR2GRAY);
  cv.GaussianBlur(imgGray, imgGray, new cv.Size(5, 5), 0);

  cv.Canny(imgGray, imgGray, 75, 200);

  const contours = new cv.MatVector();

  const hierarchy = new cv.Mat();
  cv.findContours(
    imgGray,
    contours,
    hierarchy,
    cv.RETR_CCOMP,
    cv.CHAIN_APPROX_SIMPLE
  );
  // @ts-ignore
  console.log(contours[0]);
  console.log(contours.cols);
  console.log(contours.size());
  let dst = cv.Mat.zeros(imgGray.rows, imgGray.cols, cv.CV_8UC3);

  let sortedContours: Array<SuperContour> = [];
  // @ts-ignore
  for (let i = 0; i < contours.size(); ++i) {
    // let color = new cv.Scalar(255, 0, 0);
    // cv.drawContours(dst, contours, i, color, 1, cv.LINE_8, hierarchy, 100);
    sortedContours.push({
      contour: contours.get(i),
      area: cv.contourArea(contours.get(i), true),
    });
  }

  sortedContours = sortedContours.sort((a, b) => {
    //return b.area - a.area;
    return Math.abs(b.area - a.area);
  });

  const bestcontours: Array<SuperContour> = [];

  for (let i = 0; i < 200; i++) {
    bestcontours.push(sortedContours[i]);
  }

  console.log(bestcontours);

  for (let i = 0; i < bestcontours.length; ++i) {
    let color = new cv.Scalar(255, 0, 0);
    cv.drawContours(dst, contours, i, color, 1, cv.LINE_8, hierarchy, 100);
  }

  // console.log(ctnrs.data);
  // console.log(JSON.stringify(ctnrs));
  return dst;
};

export { withEdge };
