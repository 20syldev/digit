/**
 * Neural network activation and utility functions
 */

// Rectified Linear Unit (ReLU) activation function
const relu = (x) => Math.max(0, x);

// Softmax activation function for output normalization
const softmax = (x) => {
    const sum = x.reduce((acc, val) => acc + val, 0);
    return x.map((val) => val / sum);
};

// Dense layer implementation with optional activation
const dense = (input, weights, biases, activation = relu) =>
    biases.map((bias, index) =>
        activation(
            input.reduce(
                (sum, inputVal, i) => sum + inputVal * weights[i][index],
                bias
            )
        )
    );

// Predict function chaining multiple layers
const predict = (input) => {
    let output = dense(input, W1, B1);
    output = dense(output, W2, B2);
    output = dense(output, W3, B3, Math.exp);
    return softmax(output);
};

/**
 * Image preprocessing functions
 */

// Convert image data to grayscale
const imageDataToGrayscale = (imgData) => {
    const grayscaleImg = [];
    for (let y = 0; y < imgData.height; y++) {
        grayscaleImg[y] = [];
        for (let x = 0; x < imgData.width; x++) {
            const offset = y * 4 * imgData.width + 4 * x;
            const alpha = imgData.data[offset + 3];
            grayscaleImg[y][x] = alpha === 0 ? 1 : imgData.data[offset] / 255;
        }
    }
    return grayscaleImg;
};

// Reduce image to 28x28 resolution
const reduceImage = (img) => {
    const reducedSize = 28;
    const blockSize = img.length / reducedSize;
    const reducedImg = Array.from({ length: reducedSize }, () =>
        new Array(reducedSize).fill(0)
    );

    for (let y = 0; y < reducedSize; y++) {
        for (let x = 0; x < reducedSize; x++) {
            let sum = 0;
            for (let v = 0; v < blockSize; v++) {
                for (let h = 0; h < blockSize; h++) {
                    sum += img[y * blockSize + v][x * blockSize + h];
                }
            }
            reducedImg[y][x] = 1 - sum / (blockSize * blockSize);
        }
    }
    return reducedImg;
};

// Calculate shift for centralizing the image
const getShift = (arr) => {
    const sumCoordinates = arr.reduce(
        (acc, row, x) =>
            row.reduce((rowAcc, cell, y) => {
                if (cell > 0) {
                    rowAcc.x += x;
                    rowAcc.y += y;
                    rowAcc.count++;
                }
                return rowAcc;
            }, acc),
        { x: 0, y: 0, count: 0 }
    );

    return sumCoordinates.count > 0
        ? [
                Math.floor(sumCoordinates.x / sumCoordinates.count) - arr.length / 2,
                Math.floor(sumCoordinates.y / sumCoordinates.count) - arr[0].length / 2,
            ]
        : [0, 0];
};

// Centralize the image around its center of mass
const centralize = (arr) => {
    const [dx, dy] = getShift(arr);
    return arr.map((row, x) =>
        row.map((_, y) => {
            const newX = x + dx;
            const newY = y + dy;
            return arr[newX] && arr[newX][newY] ? arr[newX][newY] : 0;
        })
    );
};

// Flatten a 2D array into a 1D array
const flatten = (arr) => arr.flat();

/**
 * Canvas setup and event listeners
 */

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button");

const CANVAS_SIZE = 280;
const CANVAS_SCALE = 1;

let isMouseDown = false;
let lastX = 0;
let lastY = 0;

// Set up canvas drawing settings
const setupCanvas = () => {
    ctx.lineWidth = 14;
    ctx.lineJoin = "round";
    ctx.font = "28px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#212121";
    ctx.strokeStyle = "#212121";
};

// Clear the canvas and reset predictions
const clearCanvas = () => {
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    for (let i = 0; i < 10; i++) {
        const element = document.getElementById(`prediction-${i}`);
        element.className = "prediction-col";
        element.children[0].children[0].style.height = "0";
    }
};

// Draw a line on the canvas
const drawLine = (fromX, fromY, toX, toY) => {
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.closePath();
    ctx.stroke();
};

// Update predictions based on the canvas drawing
const updatePredictions = async () => {
    const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const grayscaleImg = imageDataToGrayscale(imgData);
    const reducedImg = reduceImage(grayscaleImg);
    const centralizedImg = centralize(reducedImg);
    const predictions = predict(flatten(centralizedImg));
    const maxPrediction = Math.max(...predictions);

    for (let i = 0; i < predictions.length; i++) {
        const element = document.getElementById(`prediction-${i}`);
        element.children[0].children[0].style.height = `${predictions[i] * 100}%`;
        element.className =
            predictions[i] === maxPrediction
                ? "prediction-col top-prediction"
                : "prediction-col";
    }
};

// Mouse event handlers
const handleMouseDown = (event) => {
    isMouseDown = true;
    lastX = event.offsetX / CANVAS_SCALE;
    lastY = event.offsetY / CANVAS_SCALE;
};

const handleMouseMove = (event) => {
    if (!isMouseDown) return;
    drawLine(
        lastX,
        lastY,
        event.offsetX / CANVAS_SCALE,
        event.offsetY / CANVAS_SCALE
    );
    lastX = event.offsetX / CANVAS_SCALE;
    lastY = event.offsetY / CANVAS_SCALE;
};

const handleMouseUp = () => {
    isMouseDown = false;
    updatePredictions();
};

const handleMouseOut = (event) => {
    if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
        isMouseDown = false;
    }
};

// Touch event handlers
const getTouchPos = (event) => {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (event.touches[0].clientX - rect.left) / CANVAS_SCALE,
        y: (event.touches[0].clientY - rect.top) / CANVAS_SCALE,
    };
};

const handleTouchStart = (event) => {
    event.preventDefault();
    isMouseDown = true;
    const pos = getTouchPos(event);
    lastX = pos.x;
    lastY = pos.y;
};

const handleTouchMove = (event) => {
    if (!isMouseDown) return;
    event.preventDefault();
    const pos = getTouchPos(event);
    drawLine(lastX, lastY, pos.x, pos.y);
    lastX = pos.x;
    lastY = pos.y;
};

const handleTouchEnd = () => {
    isMouseDown = false;
    updatePredictions();
};

// Initialize the canvas and set event listeners
setupCanvas();

canvas.addEventListener("mousedown", handleMouseDown);
canvas.addEventListener("mousemove", handleMouseMove);
document.addEventListener("mouseup", handleMouseUp);
document.addEventListener("mouseout", handleMouseOut);
clearButton.addEventListener("click", clearCanvas);

canvas.addEventListener("touchstart", handleTouchStart);
canvas.addEventListener("touchmove", handleTouchMove);
document.addEventListener("touchend", handleTouchEnd);
