const video = document.getElementById('videoInput');
const canvasElement = document.getElementById('webcam');
const canvasCtx = canvasElement.getContext('2d');
const outputText = document.getElementById('output_text');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const loadingStatus = document.getElementById('loadingStatus');

// PASTIKAN PATH INI MENUNJUK KE MODEL GRAPH ANDA
const MODEL_PATH = './tfjs_model_graph/model.json';
const IMAGE_SIZE = 224;
const CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                     'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z', 'del', 'blank', 'space'];

let model;
let hands;
let stream;
let animationFrameId;
let isDetecting = false;

const predQueue = [];
const PRED_QUEUE_MAXLEN = 7; // Bisa disesuaikan (misal 5 atau 7)
let lastStableLabel = ""; // Label stabil terakhir yang ditambahkan ke textbox

async function loadSignLanguageModel() {
    try {
        loadingStatus.innerText = 'Memuat model deteksi isyarat...';
        console.log(`Attempting to load GraphModel from: ${MODEL_PATH}`);
        model = await tf.loadGraphModel(MODEL_PATH);

        // Warmup model
        const dummyTensor = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        let result;
        try {
            console.log("Attempting warmup with model.execute()...");
            const inputs = {};
            // Nama input 'keras_tensor_1766' dari signature model.json Anda
            inputs['keras_tensor_1766'] = dummyTensor;
            result = model.execute(inputs);
            console.log("Warmup with model.execute() likely succeeded.");
        } catch (e) {
            console.warn("Warmup with model.execute() failed, trying model.predict():", e);
            result = model.predict(dummyTensor); // Fallback
        }

        // Dispose hasil warmup
        if (Array.isArray(result)) result.forEach(t => tf.dispose(t));
        else if (result instanceof tf.Tensor) result.dispose();
        else if (typeof result === 'object' && result !== null) Object.values(result).forEach(t => { if (t instanceof tf.Tensor) tf.dispose(t); });
        tf.dispose(dummyTensor);

        loadingStatus.innerText = 'Model deteksi isyarat berhasil dimuat.';
        console.log('Sign language model (GraphModel) loaded successfully.');
        return model;
    } catch (error) {
        console.error('Fatal error loading sign language model (GraphModel):', error);
        loadingStatus.innerText = 'Gagal memuat model! Periksa konsol.';
        if (error.message) loadingStatus.innerText += ` Error: ${error.message.substring(0, 100)}...`;
        return null;
    }
}

function loadMediaPipeHands() {
    loadingStatus.innerText = 'Memuat model deteksi tangan (MediaPipe)...';
    hands = new Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1, // 0 (cepat), 1 (seimbang), 2 (akurat, lambat)
        minDetectionConfidence: 0.8, // Tingkatkan jika ada false positive (misal deteksi wajah)
        minTrackingConfidence: 0.7
    });
    hands.onResults(onHandsResults);
    loadingStatus.innerText = 'Model deteksi tangan berhasil dimuat.';
    console.log('MediaPipe Hands loaded');
}

async function startDetection() {
    if (!model) {
        model = await loadSignLanguageModel();
        if (!model) { console.error("Model loading failed. Cannot start detection."); return; }
    }
    if (!hands) loadMediaPipeHands();

    loadingStatus.innerText = 'Memulai webcam...';
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            isDetecting = true;
            startButton.disabled = true;
            stopButton.disabled = false;
            loadingStatus.innerText = 'Deteksi berjalan...';
            detectLoop();
        };
        video.onerror = (e) => { console.error("Video element error:", e); loadingStatus.innerText = "Error pada elemen video."; }
    } catch (err) {
        console.error("Error accessing webcam: ", err);
        if (err.name === "NotAllowedError") loadingStatus.innerText = 'Error: Izin webcam ditolak.';
        else if (err.name === "NotFoundError") loadingStatus.innerText = 'Error: Tidak ada webcam ditemukan.';
        else loadingStatus.innerText = 'Error: Tidak bisa mengakses webcam.';
    }
}

function stopDetection() {
    isDetecting = false;
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    if (stream) stream.getTracks().forEach(track => track.stop());
    video.srcObject = null;
    startButton.disabled = false;
    stopButton.disabled = true;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    outputText.value = "";
    predQueue.length = 0;
    lastStableLabel = "";
    loadingStatus.innerText = 'Deteksi dihentikan.';
}

async function detectLoop() {
    if (!isDetecting || video.paused || video.ended || !hands || !model || video.readyState < video.HAVE_ENOUGH_DATA) {
        if (isDetecting) animationFrameId = requestAnimationFrame(detectLoop);
        return;
    }
    try {
        await hands.send({ image: video });
    } catch (e) {
        console.error("Error sending frame to MediaPipe Hands:", e);
    }
    animationFrameId = requestAnimationFrame(detectLoop);
}

async function onHandsResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // Gambar video yang di-flip ke canvas utama untuk tampilan cermin
    canvasCtx.translate(canvasElement.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.restore();

    let labelToDrawOnFrame = "Processing..."; // Label untuk digambar di canvas per frame
    let labelForTextboxThisFrame = "";     // Label yang akan ditambahkan ke textbox untuk frame ini (jika stabil)

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0]; // Ambil tangan pertama

        // 1. Hitung Bounding Box di sistem koordinat video asli (tidak di-flip)
        let videoMinX = video.videoWidth, videoMinY = video.videoHeight, videoMaxX = 0, videoMaxY = 0;
        for (const landmark of landmarks) {
            const x = landmark.x * video.videoWidth;
            const y = landmark.y * video.videoHeight;
            videoMinX = Math.min(videoMinX, x); videoMinY = Math.min(videoMinY, y);
            videoMaxX = Math.max(videoMaxX, x); videoMaxY = Math.max(videoMaxY, y);
        }
        const margin = 25;
        const handRectXOriginal = Math.max(0, videoMinX - margin);
        const handRectYOriginal = Math.max(0, videoMinY - margin);
        const handRectWidthOriginal = Math.min(video.videoWidth - handRectXOriginal, (videoMaxX - videoMinX) + 2 * margin);
        const handRectHeightOriginal = Math.min(video.videoHeight - handRectYOriginal, (videoMaxY - videoMinY) + 2 * margin);

        // 2. Ambil ImageData dari elemen video asli menggunakan canvas offscreen (untuk input model)
        let handImageData;
        if (handRectWidthOriginal > 0 && handRectHeightOriginal > 0) {
            const offscreenCanvas = document.createElement('canvas');
            offscreenCanvas.width = handRectWidthOriginal;
            offscreenCanvas.height = handRectHeightOriginal;
            const offscreenCtx = offscreenCanvas.getContext('2d');
            offscreenCtx.drawImage(video, handRectXOriginal, handRectYOriginal, handRectWidthOriginal, handRectHeightOriginal, 0, 0, handRectWidthOriginal, handRectHeightOriginal);
            handImageData = offscreenCtx.getImageData(0, 0, handRectWidthOriginal, handRectHeightOriginal);

            // 3. Gambar Bounding Box dan Landmark yang sudah di-flip di canvas utama (untuk display)
            const flippedLandmarks = landmarks.map(lm => ({ x: 1 - lm.x, y: lm.y, z: lm.z })); // Flip koordinat x
            let displayMinX = canvasElement.width, displayMinY = canvasElement.height, displayMaxX = 0, displayMaxY = 0;
            for (const landmark of flippedLandmarks) {
                const x = landmark.x * canvasElement.width;
                const y = landmark.y * canvasElement.height;
                displayMinX = Math.min(displayMinX, x); displayMinY = Math.min(displayMinY, y);
                displayMaxX = Math.max(displayMaxX, x); displayMaxY = Math.max(displayMaxY, y);
            }
            const displayHandRectX = Math.max(0, displayMinX - margin);
            const displayHandRectY = Math.max(0, displayMinY - margin);
            const displayHandRectWidth = Math.min(canvasElement.width - displayHandRectX, (displayMaxX - displayMinX) + 2 * margin);
            const displayHandRectHeight = Math.min(canvasElement.height - displayHandRectY, (displayMaxY - displayMinY) + 2 * margin);

            if (displayHandRectWidth > 0 && displayHandRectHeight > 0) {
                canvasCtx.strokeStyle = 'lime';
                canvasCtx.lineWidth = 2;
                canvasCtx.strokeRect(displayHandRectX, displayHandRectY, displayHandRectWidth, displayHandRectHeight);
            }
            if (window.drawConnectors && window.drawLandmarks) {
                drawConnectors(canvasCtx, flippedLandmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
                drawLandmarks(canvasCtx, flippedLandmarks, { color: '#FF0000', lineWidth: 1, radius: 3 });
            }

            // --- Mulai Prediksi TF.js (Manajemen Memori Manual untuk Output Prediksi) ---
            let handTensorForModel = tf.browser.fromPixels(handImageData);
            let resizedTensorForModel = tf.image.resizeBilinear(handTensorForModel, [IMAGE_SIZE, IMAGE_SIZE]);
            let normalizedTensorForModel = resizedTensorForModel.div(tf.scalar(255.0));
            let batchedTensorForModel = normalizedTensorForModel.expandDims(0);

            // Dispose tensor input yang sudah diproses
            handTensorForModel.dispose();
            resizedTensorForModel.dispose();
            normalizedTensorForModel.dispose();

            let modelOutput;
            try {
                const inputs = {};
                inputs['keras_tensor_1766'] = batchedTensorForModel; // Nama input dari signature model.json
                modelOutput = model.execute(inputs);
            } catch (e) {
                console.warn("model.execute failed, trying model.predict:", e);
                modelOutput = model.predict(batchedTensorForModel); // Fallback
            }
            batchedTensorForModel.dispose(); // Dispose input batch setelah prediksi

            let predictionTensor; // Ini yang akan kita proses dan dispose nanti
            if (Array.isArray(modelOutput) && modelOutput.length > 0 && modelOutput[0] instanceof tf.Tensor) {
                predictionTensor = modelOutput[0];
                modelOutput.slice(1).forEach(t => { if (t instanceof tf.Tensor) t.dispose(); });
            } else if (modelOutput instanceof tf.Tensor) {
                predictionTensor = modelOutput;
            } else if (typeof modelOutput === 'object' && modelOutput !== null) {
                const outputName = 'output_0'; // Dari signature model.json
                predictionTensor = modelOutput[outputName];
                for (const key in modelOutput) {
                    if (key !== outputName && modelOutput[key] instanceof tf.Tensor) {
                        modelOutput[key].dispose();
                    }
                }
            } else {
                console.error("Unexpected model output structure:", modelOutput);
                predictionTensor = null;
            }
            // --- Akhir Bagian Prediksi TF.js ---

            if (!predictionTensor || predictionTensor.isDisposed) {
                console.error("Prediction tensor is null or disposed before processing. Skipping.");
                labelToDrawOnFrame = "Err: Pred Tensor";
            } else {
                try {
                    // Operasi data harus setelah memastikan tensor valid
                    // await dibutuhkan jika .data() dipakai, tidak jika .dataSync()
                    // const probabilities = await predictionTensor.data(); // Jika ingin akses semua probabilitas
                    const classIdTensor = predictionTensor.argMax(1);
                    const classId = classIdTensor.dataSync()[0];
                    classIdTensor.dispose();

                    if (classId === undefined || classId < 0 || classId >= CLASS_NAMES.length) {
                        console.error("Invalid Class ID from model:", classId);
                        labelToDrawOnFrame = "Err: Inv.ID";
                    } else {
                        const currentRawPrediction = CLASS_NAMES[classId];
                        predQueue.push(currentRawPrediction);
                        if (predQueue.length > PRED_QUEUE_MAXLEN) predQueue.shift();

                        labelToDrawOnFrame = currentRawPrediction; // Tampilkan prediksi mentah frame ini

                        if (predQueue.length === PRED_QUEUE_MAXLEN) {
                            const counts = {};
                            predQueue.forEach(label => counts[label] = (counts[label] || 0) + 1);
                            let maxCount = 0;
                            let majorityLabel = currentRawPrediction;
                            for (const label_key in counts) {
                                if (counts[label_key] > maxCount) {
                                    maxCount = counts[label_key];
                                    majorityLabel = label_key;
                                }
                            }
                            labelToDrawOnFrame = majorityLabel; // Update label gambar dengan mayoritas

                            // Kondisi untuk menambahkan ke textbox
                            if (maxCount >= Math.floor(PRED_QUEUE_MAXLEN * 0.7) && // Cukup stabil (misal 5 dari 7)
                                majorityLabel !== 'blank' &&
                                majorityLabel !== lastStableLabel) {
                                labelForTextboxThisFrame = majorityLabel;
                                lastStableLabel = majorityLabel; // Update label stabil terakhir yang ditambahkan
                            } else if (majorityLabel === 'blank' || maxCount < Math.floor(PRED_QUEUE_MAXLEN * 0.7) ) {
                                // Jika blank atau tidak cukup stabil, reset lastStableLabel
                                // agar prediksi valid berikutnya bisa masuk tanpa dianggap duplikat.
                                if (lastStableLabel !== "") { // Hanya reset jika sebelumnya ada label stabil non-blank
                                     lastStableLabel = "";
                                }
                            }
                        }
                    }
                } catch (e) {
                    console.error("Error processing prediction results:", e);
                    labelToDrawOnFrame = "Err: ProcPred";
                } finally {
                    if (predictionTensor && !predictionTensor.isDisposed) {
                        predictionTensor.dispose();
                    }
                }
            }
            // Gambar label prediksi frame saat ini di atas tangan (gunakan koordinat display)
            canvasCtx.fillStyle = "yellow";
            canvasCtx.font = "bold 20px Arial";
            canvasCtx.fillText(labelToDrawOnFrame, displayHandRectX, displayHandRectY - 5);

        } else { // if (handRectWidthOriginal > 0 && handRectHeightOriginal > 0)
            predQueue.length = 0;
            lastStableLabel = ""; // Reset jika tidak ada tangan/ROI valid
            labelToDrawOnFrame = "Invalid ROI";
        }
    } else { // if (results.multiHandLandmarks ...)
        labelToDrawOnFrame = "No Hand";
        predQueue.length = 0;
        lastStableLabel = ""; // Reset jika tidak ada tangan
    }

    // Tampilkan status No Hand atau label prediksi TERAKHIR yang stabil di pojok
    if (labelToDrawOnFrame === "No Hand" || labelToDrawOnFrame === "Invalid ROI" || labelToDrawOnFrame.startsWith("Err:")) {
        canvasCtx.fillStyle = "red";
        canvasCtx.font = "24px Arial";
        canvasCtx.fillText(labelToDrawOnFrame, 10, 30);
    } else if (lastStableLabel && lastStableLabel !== 'blank') { // Tampilkan lastStableLabel yang sudah valid ke textbox
        canvasCtx.fillStyle = "blue";
        canvasCtx.font = "bold 28px Arial";
        canvasCtx.fillText(`Output: ${lastStableLabel}`, 10, canvasElement.height - 20);
    }


    // Update Textbox HANYA jika labelForTextboxThisFrame memiliki nilai baru untuk frame ini
    if (labelForTextboxThisFrame) {
        if (labelForTextboxThisFrame === 'del') {
            if (outputText.value.length > 0) outputText.value = outputText.value.slice(0, -1);
        } else if (labelForTextboxThisFrame === 'space') {
            outputText.value += ' ';
        } else if (labelForTextboxThisFrame !== 'blank') { // Pastikan tidak menambah 'blank'
            outputText.value += labelForTextboxThisFrame;
        }
    }
}

startButton.addEventListener('click', startDetection);
stopButton.addEventListener('click', stopDetection);

window.addEventListener('beforeunload', () => {
    if (isDetecting) {
        stopDetection();
    }
});