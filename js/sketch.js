let x_vals = [];
let y_vals = [];

let a, b, c, d, e, f;
let dragging = false;
let degree;

const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);

function init(isReset) {
  x_vals = [];
  y_vals = [];
  degree = parseInt($('#degree_input').val());
  $('#degree_display').text('Degree = ' + degree);
  console.log("init: " + degree);
  switch(degree) {
    case 1:
      a = tf.variable(tf.scalar(random(-1, 1)));
      b = tf.variable(tf.scalar(random(-1, 1)));
      break;
    case 2:
      a = tf.variable(tf.scalar(random(-1, 1)));
      b = tf.variable(tf.scalar(random(-1, 1)));
      c = tf.variable(tf.scalar(random(-1, 1)));
      break;
    case 3:
      a = tf.variable(tf.scalar(random(-1, 1)));
      b = tf.variable(tf.scalar(random(-1, 1)));
      c = tf.variable(tf.scalar(random(-1, 1)));
      d = tf.variable(tf.scalar(random(-1, 1)));
      break;
    case 4:
      a = tf.variable(tf.scalar(random(-1, 1)));
      b = tf.variable(tf.scalar(random(-1, 1)));
      c = tf.variable(tf.scalar(random(-1, 1)));
      d = tf.variable(tf.scalar(random(-1, 1)));
      e = tf.variable(tf.scalar(random(-1, 1)));
      break;
    case 5:
      a = tf.variable(tf.scalar(random(-1, 1)));
      b = tf.variable(tf.scalar(random(-1, 1)));
      c = tf.variable(tf.scalar(random(-1, 1)));
      d = tf.variable(tf.scalar(random(-1, 1)));
      e = tf.variable(tf.scalar(random(-1, 1)));
      f = tf.variable(tf.scalar(random(-1, 1)));
      break;
  } 
}

function setup() {
  width = windowWidth / 2;
  height = windowHeight / 2;

  canvas = createCanvas(width, height);
  canvas.parent('canvas_container');
  init()
}

function loss (pred, labels) {
  return pred
    .sub(labels)
    .square()
    .mean();
}

function predict(x_vals) {
  const xs = tf.tensor1d(x_vals);
  let ys;
  console.log(degree);
  switch(degree) {
    case 1:
      ys = a.mul(xs).add(b);
      break;
    case 2:
      ys = a.mul(xs).add(b).mul(xs).add(c);
      break;
    case 3:
      ys = a.mul(xs).add(b).mul(xs).add(c).mul(xs).add(d);
      break;
    case 4:
      ys = a.mul(xs).add(b).mul(xs).add(c).mul(xs).add(d).mul(xs).add(e);
      break;
    case 5:
      ys = a.mul(xs).add(b).mul(xs).add(c).mul(xs).add(d).mul(xs).add(e).mul(xs).add(f);
      break;
    default:
      ys = a.mul(xs).add(b);
  } 
  return ys;
}

function mousePressed() {
  dragging = true;
}

function mouseReleased() {
  dragging = false;
}

function draw() {
  if (dragging) {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_vals.push(x);
    y_vals.push(y);
  } else {
    tf.tidy(() => {
      if (x_vals.length > 0) {
        const ys = tf.tensor1d(y_vals);
        optimizer.minimize(() => loss(predict(x_vals), ys));
      }
    });
  }

  background(0);

  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, height, 0);
    point(px, py);
  }

  const curveX = [];
  for (let x = -1; x <= 1; x += 0.05) {
    curveX.push(x);
  }

  const ys = tf.tidy(() => predict(curveX));
  const curveY = ys.dataSync();
  ys.dispose();

  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for (let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, height, 0);
    vertex(x, y);
  }
  endShape();

  // console.log(tf.memory().numTensors);
}

$(document).ready(function () {

  $('#degree_input').change(function () {
    init();
  });

  $('#reset_button').click(function () {
    init();
  })
});