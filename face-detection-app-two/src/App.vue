<template>
  <div class="container">
    <video ref="video" autoplay playsinline></video>
    <canvas ref="canvas"></canvas>
    <div v-if="error" class="error">{{ error }}</div>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

export default {
  name: 'App',
  data() {
    return {
      model: null,
      video: null,
      canvas: null,
      ctx: null,
      error: null,
    };
  },
 

  async mounted() {
    try {
      console.log('Attempting to set WebGL backend...');
      await tf.setBackend('webgl');

      // Check if the WebGL backend was successfully set
      if (tf.getBackend() === 'webgl') {
        console.log('WebGL backend successfully set and ready.');
      } else {
        // If not, log a specific warning and try to fall back
        console.warn('WebGL backend could not be set. Falling back to CPU.');
        await tf.setBackend('cpu');

        if (tf.getBackend() !== 'cpu') {
          throw new Error('Failed to set any TensorFlow.js backend.');
        }
      }

      await tf.ready();

      // Load the FaceMesh model
      this.model = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        {
          runtime: 'tfjs',
          refineLandmarks: true,
          maxFaces: 1,
        }
      );
      console.log('Model loaded successfully');

      // Get references to video and canvas
      this.video = this.$refs.video;
      this.canvas = this.$refs.canvas;
      this.ctx = this.canvas.getContext('2d');

      // Access webcam
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      this.video.srcObject = stream;

      // When video metadata is loaded, start detection loop
      this.video.onloadedmetadata = () => {
        this.video.play();
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        this.detectFaces();
      };
    } catch (err) {
      console.error('Initialization error:', err);
      this.error = `Failed to initialize: ${err.message}`;
    }
  },

  methods: {
    async detectFaces() {
      try {
        if (!this.model || tf.getBackend() !== 'webgl') {
          // Wait and try again
          requestAnimationFrame(() => this.detectFaces());
          return;
        }
        const predictions = await this.model.estimateFaces(this.video);

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (predictions.length > 0) {
          predictions.forEach((prediction) => {
            const keypoints = prediction.keypoints;
            const annotations = prediction.annotations;

            // Draw all landmark points
            keypoints.forEach((point) => {
            this.ctx.beginPath();
            this.ctx.arc(point[0], point[1], 2, 0, 2 * Math.PI);
            this.ctx.fillStyle = 'red';
            this.ctx.fill();
            });

            // Label key features
            const leftEyePoints = annotations.leftEyeUpper0;
            const leftEyeCenter = this.getCenter(leftEyePoints);
            this.drawLabel(leftEyeCenter.x, leftEyeCenter.y - 20, 'Left Eye');

            const rightEyePoints = annotations.rightEyeUpper0;
            const rightEyeCenter = this.getCenter(rightEyePoints);
            this.drawLabel(rightEyeCenter.x, rightEyeCenter.y - 20, 'Right Eye');

            const nosePoints = annotations.noseBottom;
            const noseCenter = this.getCenter(nosePoints);
            this.drawLabel(noseCenter.x, noseCenter.y + 10, 'Nose');

            const mouthPoints = annotations.lipsUpperOuter;
            const mouthCenter = this.getCenter(mouthPoints);
            this.drawLabel(mouthCenter.x, mouthCenter.y + 20, 'Mouth');
          });
        }

        // Continue detection loop
        requestAnimationFrame(() => this.detectFaces());
      } catch (err) {
        console.error('Detection error:', err);
        this.error = `Detection failed: ${err.message}`;
      }
    },
    getCenter(points) {
  if (!points || points.length === 0) return { x: 0, y: 0 };
  let sumX = 0, sumY = 0;
  points.forEach((p) => {
    sumX += p[0];
    sumY += p[1];
  });
  return { x: sumX / points.length, y: sumY / points.length };
},
    drawLabel(x, y, label) {
      this.ctx.font = '14px Arial';
      this.ctx.fillStyle = 'blue';
      this.ctx.fillText(label, x, y);
    },
  },
};
</script>

<style>
.container {
  position: relative;
  width: 640px;
  height: 480px;
  margin: auto;
}
video {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}
canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}
.error {
  color: red;
  text-align: center;
  margin-top: 10px;
}
</style>