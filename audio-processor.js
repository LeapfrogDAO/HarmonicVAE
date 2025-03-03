/**
 * Creates an audio processor that connects to an audio element and provides real-time analysis for visualization.
 * 
 * @param {string} audioElementId - The ID of the audio element to analyze.
 * @param {Object} [options] - Optional configuration for the audio processor.
 * @param {number} [options.fftSize=2048] - The FFT size for frequency analysis (must be a power of 2, e.g., 512, 1024, 2048).
 * @param {number} [options.smoothingTimeConstant=0.85] - Smoothing constant for the analyser (0 to 1, lower values = faster response).
 * @param {number} [options.beatThreshold=200] - Energy threshold for beat detection in bass frequencies.
 * @param {number} [options.bassBinCount=10] - Number of frequency bins analyzed for bass (beat detection).
 * @returns {Object} - Public API for audio processing and analysis.
 */
function createAudioProcessor(audioElementId, options = {}) {
  // Default configuration with fallback values
  const config = {
    fftSize: options.fftSize || 2048,
    smoothingTimeConstant: options.smoothingTimeConstant || 0.85,
    beatThreshold: options.beatThreshold || 200,
    bassBinCount: options.bassBinCount || 10,
  };

  // Validate Web Audio API support
  if (!window.AudioContext && !window.webkitAudioContext) {
    console.error('Web Audio API is not supported in this browser');
    return {
      isConnected: () => false,
      getIntensity: () => 0,
      getFrequencyData: () => new Uint8Array(0),
      getTimeData: () => new Uint8Array(0),
      detectBeat: () => false,
      getPeakFrequency: () => 0,
      getSpectralCentroid: () => 0,
      getRMS: () => 0,
      disconnect: () => {},
    };
  }

  // Initialize audio context
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContext();

  // Retrieve audio element
  const audioElement = document.getElementById(audioElementId);
  if (!audioElement) {
    console.error(`Audio element with ID ${audioElementId} not found`);
    return {
      isConnected: () => false,
      getIntensity: () => 0,
      getFrequencyData: () => new Uint8Array(0),
      getTimeData: () => new Uint8Array(0),
      detectBeat: () => false,
      getPeakFrequency: () => 0,
      getSpectralCentroid: () => 0,
      getRMS: () => 0,
      disconnect: () => {},
    };
  }

  // Internal state
  let source = null;
  let analyser = null;
  let isProcessorConnected = false;
  let animationFrame = null;
  let lastUpdateTime = 0;
  const updateFrequency = 1000 / 60; // Target 60 FPS updates

  // Data arrays for frequency and time-domain analysis
  const frequencyBinCount = config.fftSize / 2;
  const frequencyData = new Uint8Array(frequencyBinCount);
  const timeData = new Uint8Array(config.fftSize);

  // Beat detection state
  let previousBassAvg = 0;
  let beatHoldTime = 0;
  const beatHoldThreshold = 100; // Minimum time (ms) between beats

  // Connect audio processor to the audio element
  function connect() {
    if (isProcessorConnected) return;

    try {
      source = audioContext.createMediaElementSource(audioElement);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = config.fftSize;
      analyser.smoothingTimeConstant = config.smoothingTimeConstant;

      source.connect(analyser);
      analyser.connect(audioContext.destination);

      isProcessorConnected = true;
      console.log('Audio processor connected successfully');
    } catch (error) {
      console.error('Error connecting audio processor:', error);
      disconnect();
    }
  }

  // Disconnect and clean up resources
  function disconnect() {
    if (!isProcessorConnected) return;

    try {
      if (source) source.disconnect();
      if (analyser) analyser.disconnect();
      if (animationFrame) cancelAnimationFrame(animationFrame);
      isProcessorConnected = false;
      console.log('Audio processor disconnected');
    } catch (error) {
      console.error('Error disconnecting audio processor:', error);
    }
  }

  // Update frequency and time-domain data
  function updateAudioData() {
    if (!isProcessorConnected || !analyser) return;
    analyser.getByteFrequencyData(frequencyData);
    analyser.getByteTimeDomainData(timeData);
    lastUpdateTime = performance.now();
  }

  // Efficiently update data using requestAnimationFrame
  function requestUpdate() {
    const now = performance.now();
    if (now - lastUpdateTime > updateFrequency) {
      updateAudioData();
    }
    animationFrame = requestAnimationFrame(requestUpdate);
  }

  // Calculate overall audio intensity (normalized 0-1)
  function getIntensity() {
    if (!isProcessorConnected) return 0;
    updateAudioData();
    const sum = frequencyData.reduce((acc, val) => acc + val, 0);
    return sum / (frequencyData.length * 255);
  }

  // Enhanced beat detection based on bass energy peaks
  function detectBeat() {
    if (!isProcessorConnected) return false;
    updateAudioData();

    const bassRange = frequencyData.slice(0, config.bassBinCount);
    const bassAvg = bassRange.reduce((acc, val) => acc + val, 0) / config.bassBinCount;

    // Detect significant bass energy increase
    const isBeat = bassAvg > config.beatThreshold && bassAvg > previousBassAvg * 1.2;
    previousBassAvg = bassAvg;

    // Prevent rapid consecutive beats
    const now = performance.now();
    if (isBeat && now - beatHoldTime > beatHoldThreshold) {
      beatHoldTime = now;
      return true;
    }
    return false;
  }

  // Get frequency with the highest magnitude
  function getPeakFrequency() {
    if (!isProcessorConnected) return 0;
    updateAudioData();
    const maxIndex = frequencyData.indexOf(Math.max(...frequencyData));
    const nyquist = audioContext.sampleRate / 2;
    return (maxIndex / frequencyData.length) * nyquist;
  }

  // Calculate spectral centroid (indicates sound "brightness")
  function getSpectralCentroid() {
    if (!isProcessorConnected) return 0;
    updateAudioData();
    let sumAmplitude = 0;
    let weightedSum = 0;
    for (let i = 0; i < frequencyData.length; i++) {
      sumAmplitude += frequencyData[i];
      weightedSum += i * frequencyData[i];
    }
    return sumAmplitude > 0 ? weightedSum / sumAmplitude : 0;
  }

  // Calculate RMS (root mean square) for volume
  function getRMS() {
    if (!isProcessorConnected) return 0;
    updateAudioData();
    let sum = 0;
    for (let i = 0; i < timeData.length; i++) {
      const sample = (timeData[i] - 128) / 128; // Normalize around zero
      sum += sample * sample;
    }
    return Math.sqrt(sum / timeData.length);
  }

  // Set up event listeners for audio element
  function initialize() {
    audioElement.addEventListener('play', () => {
      if (audioContext.state === 'suspended') audioContext.resume();
      connect();
      requestUpdate(); // Start continuous updates
    });
    audioElement.addEventListener('pause', disconnect);
    audioElement.addEventListener('ended', disconnect);
  }

  // Initialize the processor
  initialize();

  // Public API
  return {
    /** @returns {boolean} Whether the processor is connected */
    isConnected: () => isProcessorConnected,
    /** @returns {number} Normalized audio intensity (0-1) */
    getIntensity: () => getIntensity(),
    /** @returns {Uint8Array} Frequency data for visualization */
    getFrequencyData: () => frequencyData,
    /** @returns {Uint8Array} Time-domain data for waveform visualization */
    getTimeData: () => timeData,
    /** @returns {boolean} Whether a beat is detected */
    detectBeat: () => detectBeat(),
    /** @returns {number} Frequency with the highest magnitude (Hz) */
    getPeakFrequency: () => getPeakFrequency(),
    /** @returns {number} Spectral centroid (indicative of sound brightness) */
    getSpectralCentroid: () => getSpectralCentroid(),
    /** @returns {number} Root mean square value (volume level) */
    getRMS: () => getRMS(),
    /** Disconnects the processor and cleans up resources */
    disconnect: () => disconnect(),
  };
}
