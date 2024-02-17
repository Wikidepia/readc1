import 'jimp';
const ort = require('onnxruntime-web');

async function letterboxImage(image, size) {
    let insideImg = await image.clone();
    const iw = insideImg.bitmap.width;
    const ih = insideImg.bitmap.height;
    const [w, h] = size;
    const scale = Math.min(w / iw, h / ih);
    const nw = Math.floor(iw * scale);
    const nh = Math.floor(ih * scale);
  
    insideImg = await insideImg.resize(nw, nh, Jimp.RESIZE_BICUBIC);
    const newImage = new Jimp(w, h, 0x727272ff);
    newImage.composite(insideImg, (w - nw) / 2, (h - nh) / 2);
    return newImage;
  }

  function scaleBoxes(boxes, imageDims, scaledDims) {
    const [iw, ih] = imageDims;
    const [sw, sh] = scaledDims;
    const scale = Math.min(sw / iw, sh / ih);
    const nw = Math.floor(iw * scale);
    const nh = Math.floor(ih * scale);

    const [x, y, w, h] = boxes;
    const xs = (x - (sw - nw) / 2) / scale;
    const ys = (y - (sh - nh) / 2) / scale;
    const ws = w / scale;
    const hs = h / scale;
    return [xs, ys, ws, hs];
  }
  
  function imageDataToTensor(image, dims, normalize = true) {
    // 1. Get buffer data from image and extract R, G, and B arrays.
    var imageBufferData = image.bitmap.data;
    const [redArray, greenArray, blueArray] = [[], [], []];
  
    // 2. Loop through the image buffer and extract the R, G, and B channels
    for (let i = 0; i < imageBufferData.length; i += 4) {
      redArray.push(imageBufferData[i]);
      greenArray.push(imageBufferData[i + 1]);
      blueArray.push(imageBufferData[i + 2]);
    }
  
    // 3. Concatenate RGB to transpose [256, 256, 3] -> [3, 256, 256] to a number array
    const transposedData = redArray.concat(greenArray, blueArray);
  
    // 4. Convert to float32 and normalize to 1
    const float32Data = new Float32Array(transposedData.map((x) => x / 255.0));
  
    // 5. Normalize the data mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    if (normalize) {
      const mean = [0.485, 0.456, 0.406];
      const std = [0.229, 0.224, 0.225];
      for (let i = 0; i < float32Data.length; i++) {
        float32Data[i] = (float32Data[i] - mean[i % 3]) / std[i % 3];
      }
    }

    // 6. Create a tensor from the float32 data
    const inputTensor = new ort.Tensor('float32', float32Data, dims);
    return inputTensor;
  }

async function runNumbox(image, numboxSession, nmsSession) {
    const imagex = await image.clone()
    const iw = imagex.bitmap.width;
    const ih = imagex.bitmap.height;
    imagex.rgba(false);

    const numboxInputImage = await letterboxImage(imagex, [640, 640]);
    const numboxInputTensor = imageDataToTensor(
        numboxInputImage,
        [1, 3, 640, 640],
        false
    );

    const config = new ort.Tensor(
        'float32',
        new Float32Array([3, 0.25, 0.6])
    );
    const numboxOutputMap = await numboxSession.run({ images: numboxInputTensor });
    const numboxOutput = numboxOutputMap[numboxSession.outputNames[0]];
    const nmsOutput = await nmsSession.run({
        detection: numboxOutputMap[numboxSession.outputNames[0]],
        config: config,
    });
    const selectedIdx = nmsOutput[nmsSession.outputNames[0]];

    const boxes = [];
    for (let i = 0; i < selectedIdx.data.length; i++) {
        const idx = selectedIdx.data[i];
        const data = numboxOutput.data.slice(
            idx * numboxOutput.dims[2],
            (idx + 1) * numboxOutput.dims[2]
        );
        const [x, y, w, h] = data.slice(0, 4);
        const [xs, ys, ws, hs] = scaleBoxes(
            [x, y, w, h],
            [iw, ih],
            [640, 640]
        );
        boxes.push([xs, ys, ws, hs]);
    }
    return boxes;
}

async function appendImageToGrid(image, gridId) {
    const grid = document.getElementById(gridId);
    const divElement = document.createElement("div");
    const imgElement = document.createElement("img");
    imgElement.src = await image.getBase64Async(Jimp.MIME_PNG);
    divElement.appendChild(imgElement);
    grid.appendChild(divElement);
    return imgElement.src;
}

export default async function main(files) {
    const imageEl = document.getElementById("form-image");
    imageEl.src = URL.createObjectURL(files[0]);

    const localizerSession = await ort.InferenceSession.create(
        "./models/localizer-yolov5n.onnx"
    );
    const numboxSession = await ort.InferenceSession.create(
        "./models/numboxer-yolov5n.onnx"
    );
    const nmsSession = await ort.InferenceSession.create(
        "./models/nms-yolov5.ort"
    );

    const image = await Jimp.read(imageEl.src);
    const iw = image.bitmap.width;
    const ih = image.bitmap.height;
    image.rgba(false);

    // [topK, ioUThreshold, scoreThreshold]
    const config = new ort.Tensor(
        'float32',
        new Float32Array([10, 0.25, 0.6])
    );
    const inputImage = await letterboxImage(image, [640, 640]);
    const inputTensor = imageDataToTensor(
        inputImage,
        [1, 3, 640, 640],
        false
    );

    // YOLOv5 detector
    const outputMap = await localizerSession.run({ images: inputTensor });
    const output0 = outputMap[localizerSession.outputNames[0]];

    // NMS (Non-Maximum Suppression)
    const nmsOutput = await nmsSession.run({
        detection: outputMap[localizerSession.outputNames[0]],
        config: config,
    });
    const selectedIdx = nmsOutput[nmsSession.outputNames[0]];

    for (let i = 0; i < selectedIdx.data.length; i++) {
        const idx = selectedIdx.data[i];
        const data = output0.data.slice(
            idx * output0.dims[2],
            (idx + 1) * output0.dims[2]
        );
        // Skip if class is not numbox
        const cls = Math.round(data[6]);
        if (cls !== 0)
            continue;
        
        const [x, y, w, h] = data.slice(0, 4);        
        const [xs, ys, ws, hs] = scaleBoxes(
            [x, y, w, h],
            [iw, ih],
            [640, 640]
        );
        const [x1, y1, x2, y2] = [
            xs - ws / 2,
            ys - hs / 2,
            xs + ws / 2,
            ys + hs / 2,
        ];

        // Crop the detected object
        const ximg = await Jimp.read(imageEl.src);
        const localizedImage = ximg.crop(
            x1, y1, x2 - x1, y2 - y1
        );

        // Display the cropped image
        const localImgSrc = await appendImageToGrid(localizedImage, "localization-grid");

        // Run Numbox
        let numboxes = [];
        const boxes = await runNumbox(localizedImage, numboxSession, nmsSession);
        for (let j = 0; j < boxes.length; j++) {
            const [x, y, w, h] = boxes[j];
            const [x1, y1, x2, y2] = [
                x - w / 2,
                y - h / 2,
                x + w / 2,
                y + h / 2,
            ];
            numboxes.push([x1, y1, x2 - x1, y2 - y1]);
        }

        // Sort numboxes based on x1
        numboxes.sort((a, b) => a[0] - b[0]);
        for (let j = 0; j < numboxes.length; j++) {
            const [x, y, w, h] = numboxes[j];
            const localImg = await Jimp.read(localImgSrc);
            const croppedImage = localImg.crop(x,y,w,h);
            await appendImageToGrid(croppedImage, "numbox-grid");
        }
    }
}
