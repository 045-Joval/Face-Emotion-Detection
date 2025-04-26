import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';

const EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
const IMAGE_SIZE = 48;

const EmotionDetector = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const blazefaceModelRef = useRef(null);
    const emotionModelRef = useRef(null);
    const animationFrameIdRef = useRef(null);
    const streamRef = useRef(null);

    const [status, setStatus] = useState('Loading models...');
    const [detectedFaces, setDetectedFaces] = useState([]);
    const [isMobile, setIsMobile] = useState(false);

    const loadModels = useCallback(async () => {
        try {
            blazefaceModelRef.current = await blazeface.load();
            emotionModelRef.current = await tf.loadLayersModel('/face_emotion_model_browser/model.json');
            setStatus('Models loaded. Setting up webcam...');
            setupWebcam();
        } catch (err) {
            setStatus(`Error: ${err.message}`);
        }
    }, []);

    const setupWebcam = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;

                videoRef.current.onloadedmetadata = () => {
                    videoRef.current.play().then(() => {
                        const videoWidth = videoRef.current.videoWidth;
                        const videoHeight = videoRef.current.videoHeight;

                        if (canvasRef.current) {
                            canvasRef.current.width = videoWidth;
                            canvasRef.current.height = videoHeight;
                            canvasRef.current.style.width = `${videoWidth}px`;
                            canvasRef.current.style.height = `${videoHeight}px`;
                        }

                        setStatus('Ready');
                        detectFaces();
                    }).catch(err => {
                        setStatus(`Error playing video: ${err.message}`);
                    });
                };
            }
        } catch (err) {
            setStatus(`Camera error: ${err.message}`);
        }
    }, []);

    const detectFaces = useCallback(async () => {
        if (!videoRef.current || !canvasRef.current ||
            !blazefaceModelRef.current || !emotionModelRef.current) {
            animationFrameIdRef.current = requestAnimationFrame(detectFaces);
            return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (video.readyState !== 4 || video.paused || video.ended) {
            animationFrameIdRef.current = requestAnimationFrame(detectFaces);
            return;
        }

        try {
            const returnTensors = false;
            const faces = await blazefaceModelRef.current.estimateFaces(video, returnTensors);
            const facesWithEmotions = [];

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            for (const face of faces) {
                const [x1, y1] = face.topLeft;
                const [x2, y2] = face.bottomRight;

                const faceBox = {
                    x: x1,
                    y: y1,
                    width: x2 - x1 + 20,
                    height: y2 - y1 + 20
                };

                const result = await processFace(video, faceBox);
                if (!result) continue;

                facesWithEmotions.push(result);

                const label = `${result.dominantEmotion.emotion} (${(result.dominantEmotion.probability * 100).toFixed(1)}%)`;

                ctx.strokeStyle = 'yellow';
                ctx.lineWidth = 2;
                ctx.strokeRect(faceBox.x, faceBox.y, faceBox.width, faceBox.height);

                ctx.font = '16px Arial';
                const textWidth = ctx.measureText(label).width;
                ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                ctx.fillRect(faceBox.x, faceBox.y - 20, textWidth + 10, 20);

                ctx.fillStyle = 'yellow';
                ctx.fillText(label, faceBox.x + 5, faceBox.y - 5);
            }

            setDetectedFaces(facesWithEmotions);
        } catch (err) {
            console.error('Detection error:', err);
        }

        animationFrameIdRef.current = requestAnimationFrame(detectFaces);
    }, []);

    const processFace = useCallback((video, faceBox) => {
        return tf.tidy(() => {
            const videoHeight = video.videoHeight;
            const videoWidth = video.videoWidth;

            const y1 = faceBox.y / videoHeight;
            const x1 = faceBox.x / videoWidth;
            const y2 = (faceBox.y + faceBox.height) / videoHeight;
            const x2 = (faceBox.x + faceBox.width) / videoWidth;

            const box = [[
                Math.max(0, y1 - 0.03),
                Math.max(0, x1 - 0.03),
                Math.min(1, y2 + 0.03),
                Math.min(1, x2 + 0.03)
            ]];

            const frame = tf.browser.fromPixels(video);
            if (frame.shape[0] === 0 || frame.shape[1] === 0) {
                frame.dispose();
                return null;
            }

            const face = tf.image.cropAndResize(
                frame.expandDims(0),
                box,
                [0],
                [IMAGE_SIZE, IMAGE_SIZE]
            );

            const grayscale = face.mean(3).expandDims(-1);
            const normalized = grayscale.div(255.0);
            const prediction = emotionModelRef.current.predict(normalized);
            const probs = prediction.dataSync();

            const emotions = EMOTION_LABELS.map((emotion, i) => ({
                emotion,
                probability: probs[i]
            }));

            const dominantEmotion = emotions.reduce((max, curr) =>
                curr.probability > max.probability ? curr : max
            );

            return {
                box: faceBox,
                emotions,
                dominantEmotion
            };
        });
    }, []);

    useEffect(() => {
        tf.env().set('WEBGL_CPU_FORWARD', false);
        tf.env().set('WEBGL_PACK', true);

        setIsMobile(window.innerWidth < 768); // Detect mobile
        loadModels();

        return () => {
            if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
            try { blazefaceModelRef.current?.dispose?.(); } catch (e) {}
            try { emotionModelRef.current?.dispose?.(); } catch (e) {}
            try { tf.disposeVariables(); } catch (e) {}
        };
    }, [loadModels]);

    return (
        <div style={{ padding: '10px', maxWidth: '800px', margin: '0 auto' }}>
            <h2 style={{ textAlign: 'center', marginBottom: '20px' }}>Face Emotion Detection</h2>

            <div style={{ position: 'relative' }}>
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    style={{
                        width: '100%',
                        borderRadius: '12px',
                        objectFit: 'cover'
                    }}
                />
                <canvas
                    ref={canvasRef}
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        zIndex: 10,
                        pointerEvents: 'none'
                    }}
                />
            </div>

            <div style={{ marginTop: '20px', textAlign: 'center' }}>
                {status.startsWith('Loading') && (
                    <div className="loader"></div>
                )}
                <p>Status: {status}</p>
            </div>

            {!isMobile && detectedFaces.map((face, index) => (
                <div
                    key={index}
                    style={{
                        position: 'absolute',
                        top: face.box.y - 100,
                        left: face.box.x + 450,
                        background: 'rgba(0, 0, 0, 0.7)',
                        color: 'white',
                        padding: '10px',
                        borderRadius: '8px',
                        fontSize: '12px',
                        width: `${Math.min(face.box.width, 200)}px`,
                        zIndex: 5,
                        animation: 'float 3s ease-in-out infinite'
                    }}
                >
                    <strong>{face.dominantEmotion.emotion} ({(face.dominantEmotion.probability * 100).toFixed(1)}%)</strong>
                    {face.emotions.map((e, idx) => (
                        <div key={idx} style={{ marginBottom: '4px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>{e.emotion}</span>
                                <span>{(e.probability * 100).toFixed(1)}%</span>
                            </div>
                            <div style={{ background: '#555', borderRadius: '4px', overflow: 'hidden' }}>
                                <div style={{
                                    width: `${(e.probability * 100).toFixed(1)}%`,
                                    background: '#4caf50',
                                    height: '6px'
                                }} />
                            </div>
                        </div>
                    ))}
                </div>
            ))}

            <style>{`
                .loader {
                    border: 6px solid #f3f3f3;
                    border-top: 6px solid #4caf50;
                    border-radius: 50%;
                    width: 30px;
                    height: 30px;
                    animation: spin 1s linear infinite;
                    margin: 0 auto 10px;
                }

                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }

                @keyframes float {
                    0%, 100% { transform: translateY(0px); }
                    50% { transform: translateY(-10px); }
                }
            `}</style>
        </div>
    );
};

export default EmotionDetector;
