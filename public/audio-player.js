/**
 * audio-player.js
 * A custom audio player component for the HarmonicVAE Feedback System
 * Includes audio visualization and analysis features
 */

class HarmonicAudioPlayer {
  /**
   * Constructor for the HarmonicAudioPlayer
   * @param {Object} config - Configuration object
   * @param {string} config.containerId - ID of the container element
   * @param {string} config.audioPath - Path to the initial audio file
   * @param {boolean} config.autoplay - Whether to autoplay the audio (default: false)
   * @param {boolean} config.loop - Whether to loop the audio (default: false)
   * @param {boolean} config.showVisualizer - Whether to show the audio visualizer (default: true)
   * @param {string} config.visualizerType - Type of visualizer ('waveform', 'frequency', 'circular')
   * @param {Function} config.onPlay - Callback when audio starts playing
   * @param {Function} config.onPause - Callback when audio is paused
   * @param {Function} config.onEnded - Callback when audio ends
   * @param {Function} config.onTimeUpdate - Callback when audio time updates
   */
  constructor(config) {
    this.config = {
      containerId: 'audio-player-container',
      audioPath: null,
      autoplay: false,
      loop: false,
      showVisualizer: true,
      visualizerType: 'waveform',
      onPlay: () => {},
      onPause: () => {},
      onEnded: () => {},
      onTimeUpdate: () => {},
      ...config
    };

    this.isPlaying = false;
    this.duration = 0;
    this.currentTime = 0;
    this.audioContext = null;
    this.analyser = null;
    this.dataArray = null;
    this.audioBuffer = null;
    this.sourceNode = null;
    this.animationFrame = null;

    this._init();
  }

  /**
   * Initialize the audio player
   * @private
   */
  _init() {
    // Get container
    this.container = document.getElementById(this.config.containerId);
    if (!this.container) {
      console.error(`Container with ID '${this.config.containerId}' not found`);
      return;
    }

    // Create audio context
    try {
      window.AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioContext();
    } catch (e) {
      console.error('Web Audio API is not supported in this browser', e);
      return;
    }

    // Create HTML structure
    this._createPlayerElements();

    // Setup audio analyzer
    this._setupAudioAnalyzer();

    // Load initial audio if provided
    if (this.config.audioPath) {
      this.loadAudio(this.config.audioPath);
    }

    // Set up event handlers
    this._setupEventHandlers();

    // Make the player accessible globally
    if (!window.harmonicVAE) {
      window.harmonicVAE = {};
    }
    window.harmonicVAE.audioPlayer = this;
    window.harmonicVAE.audioAnalyzer = {
      getFrequencyData: this.getFrequencyData.bind(this),
      getWaveformData: this.getWaveformData.bind(this),
      isPlaying: () => this.isPlaying
    };
  }

  /**
   * Create player HTML elements
   * @private
   */
  _createPlayerElements() {
    this.container.innerHTML = '';
    this.container.classList.add('harmonic-audio-player');

    // Create player wrapper
    const playerWrapper = document.createElement('div');
    playerWrapper.className = 'player-wrapper';
    
    // Create audio element
    this.audioElement = document.createElement('audio');
    this.audioElement.className = 'audio-element';
    this.audioElement.setAttribute('preload', 'auto');
    
    // Create player controls
    const controlsWrapper = document.createElement('div');
    controlsWrapper.className = 'controls-wrapper';
    
    // Play/pause button
    this.playButton = document.createElement('button');
    this.playButton.className = 'play-button';
    this.playButton.innerHTML = '<i class="fas fa-play"></i>';
    this.playButton.setAttribute('aria-label', 'Play');
    
    // Progress bar
    const progressWrapper = document.createElement('div');
    progressWrapper.className = 'progress-wrapper';
    
    this.progressBar = document.createElement('div');
    this.progressBar.className = 'progress-bar';
    
    this.progressFill = document.createElement('div');
    this.progressFill.className = 'progress-fill';
    
    this.progressBar.appendChild(this.progressFill);
    progressWrapper.appendChild(this.progressBar);
    
    // Time display
    this.timeDisplay = document.createElement('div');
    this.timeDisplay.className = 'time-display';
    this.timeDisplay.textContent = '0:00 / 0:00';
    
    // Volume control
    const volumeWrapper = document.createElement('div');
    volumeWrapper.className = 'volume-wrapper';
    
    this.volumeIcon = document.createElement('button');
    this.volumeIcon.className = 'volume-icon';
    this.volumeIcon.innerHTML = '<i class="fas fa-volume-up"></i>';
    this.volumeIcon.setAttribute('aria-label', 'Mute');
    
    this.volumeSlider = document.createElement('input');
    this.volumeSlider.type = 'range';
    this.volumeSlider.min = 0;
    this.volumeSlider.max = 1;
    this.volumeSlider.step = 0.01;
    this.volumeSlider.value = 1;
    this.volumeSlider.className = 'volume-slider';
    
    volumeWrapper.appendChild(this.volumeIcon);
    volumeWrapper.appendChild(this.volumeSlider);
    
    // Assemble controls
    controlsWrapper.appendChild(this.playButton);
    controlsWrapper.appendChild(progressWrapper);
    controlsWrapper.appendChild(this.timeDisplay);
    controlsWrapper.appendChild(volumeWrapper);
    
    // Create visualizer canvas if enabled
    if (this.config.showVisualizer) {
      this.visualizerContainer = document.createElement('div');
      this.visualizerContainer.className = 'visualizer-container';
      
      this.canvas = document.createElement('canvas');
      this.canvas.className = 'visualizer-canvas';
      this.canvas.width = this.container.clientWidth;
      this.canvas.height = 80;
      
      this.visualizerContainer.appendChild(this.canvas);
      this.canvasContext = this.canvas.getContext('2d');
    }
    
    // Assemble player
    playerWrapper.appendChild(this.audioElement);
    playerWrapper.appendChild(controlsWrapper);
    
    this.container.appendChild(playerWrapper);
    
    if (this.config.showVisualizer) {
      this.container.appendChild(this.visualizerContainer);
    }
    
    // Add styles
    this._addStyles();
  }

  /**
   * Add CSS styles for the player
   * @private
   */
  _addStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .harmonic-audio-player {
        width: 100%;
        background-color: #2c3e50;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        color: #ecf0f1;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      }
      
      .player-wrapper {
        width: 100%;
      }
      
      .controls-wrapper {
        display: flex;
        align-items: center;
        padding: 10px 0;
      }
      
      .play-button {
        background-color: #3498db;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        margin-right: 15px;
        color: white;
        transition: background-color 0.2s;
      }
      
      .play-button:hover {
        background-color: #2980b9;
      }
      
      .progress-wrapper {
        flex-grow: 1;
        margin: 0 15px;
      }
      
      .progress-bar {
        width: 100%;
        height: 8px;
        background-color: #34495e;
        border-radius: 4px;
        overflow: hidden;
        cursor: pointer;
      }
      
      .progress-fill {
        width: 0%;
        height: 100%;
        background-color: #e74c3c;
        border-radius: 4px;
        transition: width 0.1s;
      }
      
      .time-display {
        min-width: 90px;
        text-align: center;
        font-size: 14px;
      }
      
      .volume-wrapper {
        display: flex;
        align-items: center;
        margin-left: 15px;
      }
      
      .volume-icon {
        background: none;
        border: none;
        color: #ecf0f1;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 5px;
      }
      
      .volume-slider {
        width: 80px;
        cursor: pointer;
      }
      
      .visualizer-container {
        margin-top: 10px;
        width: 100%;
        height: 80px;
        background-color: #34495e;
        border-radius: 4px;
        overflow: hidden;
      }
      
      .visualizer-canvas {
        width: 100%;
        height: 100%;
      }
    `;
    
    document.head.appendChild(style);
  }

  /**
   * Set up audio analyzer for visualizations
   * @private
   */
  _setupAudioAnalyzer() {
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 2048;
    this.bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(this.bufferLength);
  }

  /**
   * Set up event handlers
   * @private
   */
  _setupEventHandlers() {
    // Play/pause button
    this.playButton.addEventListener('click', () => {
      if (this.isPlaying) {
        this.pause();
      } else {
        this.play();
      }
    });
    
    // Progress bar
    this.progressBar.addEventListener('click', (e) => {
      const rect = this.progressBar.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = x / rect.width;
      const seekTime = percentage * this.duration;
      this.seek(seekTime);
    });
    
    // Volume controls
    this.volumeIcon.addEventListener('click', () => {
      if (this.audioElement.muted) {
        this.audioElement.muted = false;
        this.volumeIcon.innerHTML = '<i class="fas fa-volume-up"></i>';
      } else {
        this.audioElement.muted = true;
        this.volumeIcon.innerHTML = '<i class="fas fa-volume-mute"></i>';
      }
    });
    
    this.volumeSlider.addEventListener('input', () => {
      const volume = parseFloat(this.volumeSlider.value);
      this.setVolume(volume);
      
      // Update volume icon
      if (volume === 0) {
        this.volumeIcon.innerHTML = '<i class="fas fa-volume-mute"></i>';
      } else if (volume < 0.5) {
        this.volumeIcon.innerHTML = '<i class="fas fa-volume-down"></i>';
      } else {
        this.volumeIcon.innerHTML = '<i class="fas fa-volume-up"></i>';
      }
    });
    
    // Audio element events
    this.audioElement.addEventListener('loadedmetadata', () => {
      this.duration = this.audioElement.duration;
      this._updateTimeDisplay();
    });
    
    this.audioElement.addEventListener('timeupdate', () => {
      this.currentTime = this.audioElement.currentTime;
      this._updateProgressBar();
      this._updateTimeDisplay();
      this.config.onTimeUpdate(this.currentTime, this.duration);
    });
    
    this.audioElement.addEventListener('ended', () => {
      this.isPlaying = false;
      this.playButton.innerHTML = '<i class="fas fa-play"></i>';
      this.playButton.setAttribute('aria-label', 'Play');
      
      if (this.animationFrame) {
        cancelAnimationFrame(this.animationFrame);
        this.animationFrame = null;
      }
      
      this.config.onEnded();
    });
    
    // Window resize
    window.addEventListener('resize', () => {
      if (this.canvas) {
        this.canvas.width = this.container.clientWidth;
      }
    });
  }

  /**
   * Update the progress bar
   * @private
   */
  _updateProgressBar() {
    const percentage = (this.currentTime / this.duration) * 100;
    this.progressFill.style.width = `${percentage}%`;
  }

  /**
   * Update the time display
   * @private
   */
  _updateTimeDisplay() {
    const formatTime = (time) => {
      const minutes = Math.floor(time / 60);
      const seconds = Math.floor(time % 60);
      return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    };
    
    this.timeDisplay.textContent = `${formatTime(this.currentTime)} / ${formatTime(this.duration)}`;
  }

  /**
   * Render audio visualizer
   * @private
   */
  _renderVisualizer() {
    if (!this.canvasContext || !this.isPlaying) return;
    
    this.analyser.getByteFrequencyData(this.dataArray);
    
    const width = this.canvas.width;
    const height = this.canvas.height;
    
    this.canvasContext.clearRect(0, 0, width, height);
    
    if (this.config.visualizerType === 'waveform') {
      this._renderWaveform(width, height);
    } else if (this.config.visualizerType === 'frequency') {
      this._renderFrequency(width, height);
    } else if (this.config.visualizerType === 'circular') {
      this._renderCircular(width, height);
    }
    
    this.animationFrame = requestAnimationFrame(() => this._renderVisualizer());
  }

  /**
   * Render waveform visualizer
   * @param {number} width - Canvas width
   * @param {number} height - Canvas height
   * @private
   */
  _renderWaveform(width, height) {
    this.analyser.getByteTimeDomainData(this.dataArray);
    
    this.canvasContext.lineWidth = 2;
    this.canvasContext.strokeStyle = '#e74c3c';
    this.canvasContext.beginPath();
    
    const sliceWidth = width / this.bufferLength;
    let x = 0;
    
    for (let i = 0; i < this.bufferLength; i++) {
      const v = this.dataArray[i] / 128.0;
      const y = v * height / 2;
      
      if (i === 0) {
        this.canvasContext.moveTo(x, y);
      } else {
        this.canvasContext.lineTo(x, y);
      }
      
      x += sliceWidth;
    }
    
    this.canvasContext.lineTo(width, height / 2);
    this.canvasContext.stroke();
  }

  /**
   * Render frequency visualizer
   * @param {number} width - Canvas width
   * @param {number} height - Canvas height
   * @private
   */
  _renderFrequency(width, height) {
    this.analyser.getByteFrequencyData(this.dataArray);
    
    const barWidth = (width / this.bufferLength) * 2.5;
    let x = 0;
    
    for (let i = 0; i < this.bufferLength; i++) {
      const barHeight = this.dataArray[i] / 255 * height;
      
      const gradient = this.canvasContext.createLinearGradient(0, height, 0, height - barHeight);
      gradient.addColorStop(0, '#e74c3c');
      gradient.addColorStop(0.5, '#f39c12');
      gradient.addColorStop(1, '#f1c40f');
      
      this.canvasContext.fillStyle = gradient;
      this.canvasContext.fillRect(x, height - barHeight, barWidth, barHeight);
      
      x += barWidth + 1;
      if (x > width) break;
    }
  }

  /**
   * Render circular visualizer
   * @param {number} width - Canvas width
   * @param {number} height - Canvas height
   * @private
   */
  _renderCircular(width, height) {
    this.analyser.getByteFrequencyData(this.dataArray);
    
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(centerX, centerY) - 5;
    
    this.canvasContext.save();
    this.canvasContext.translate(centerX, centerY);
    
    for (let i = 0; i < this.bufferLength; i++) {
      const angle = (i * 2 * Math.PI) / this.bufferLength;
      const barHeight = (this.dataArray[i] / 255) * radius * 0.5;
      
      const hue = i / this.bufferLength * 360;
      this.canvasContext.fillStyle = `hsl(${hue}, 80%, 50%)`;
      
      this.canvasContext.rotate(angle);
      this.canvasContext.fillRect(0, radius - barHeight, 2, barHeight);
      this.canvasContext.rotate(-angle);
    }
    
    this.canvasContext.restore();
  }

  /**
   * Public API methods
   */

  /**
   * Load audio from URL
   * @param {string} url - URL of the audio file
   * @returns {Promise} Promise that resolves when audio is loaded
   */
  loadAudio(url) {
    return new Promise((resolve, reject) => {
      this.audioElement.src = url;
      this.audioElement.load();
      
      this.audioElement.addEventListener('canplaythrough', () => {
        resolve();
      }, { once: true });
      
      this.audioElement.addEventListener('error', (err) => {
        reject(err);
      }, { once: true });
      
      // Set autoplay and loop based on config
      this.audioElement.autoplay = this.config.autoplay;
      this.audioElement.loop = this.config.loop;
      
      // Update isPlaying if autoplay is enabled
      if (this.config.autoplay) {
        this.isPlaying = true;
        this.playButton.innerHTML = '<i class="fas fa-pause"></i>';
        this.playButton.setAttribute('aria-label', 'Pause');
      }
    });
  }

  /**
   * Play audio
   */
  play() {
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    
    if (!this.sourceNode) {
      this.sourceNode = this.audioContext.createMediaElementSource(this.audioElement);
      this.sourceNode.connect(this.analyser);
      this.analyser.connect(this.audioContext.destination);
    }
    
    this.audioElement.play();
    this.isPlaying = true;
    this.playButton.innerHTML = '<i class="fas fa-pause"></i>';
    this.playButton.setAttribute('aria-label', 'Pause');
    
    // Start visualizer
    if (this.config.showVisualizer && !this.animationFrame) {
      this._renderVisualizer();
    }
    
    // Call onPlay callback
    this.config.onPlay();
  }

  /**
   * Pause audio
   */
  pause() {
    this.audioElement.pause();
    this.isPlaying = false;
    this.playButton.innerHTML = '<i class="fas fa-play"></i>';
    this.playButton.setAttribute('aria-label', 'Play');
    
    // Stop visualizer
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    
    // Call onPause callback
    this.config.onPause();
  }

  /**
   * Stop audio (pause and reset position)
   */
  stop() {
    this.pause();
    this.seek(0);
  }

  /**
   * Seek to specific time
   * @param {number} time - Time in seconds
   */
  seek(time) {
    this.audioElement.currentTime = Math.max(0, Math.min(time, this.duration));
  }

  /**
   * Set volume
   * @param {number} volume - Volume level (0-1)
   */
  setVolume(volume) {
    this.audioElement.volume = Math.max(0, Math.min(1, volume));
    this.volumeSlider.value = this.audioElement.volume;
  }

  /**
   * Set loop
   * @param {boolean} loop - Whether audio should loop
   */
  setLoop(loop) {
    this.audioElement.loop = loop;
    this.config.loop = loop;
  }

  /**
   * Set visualizer type
   * @param {string} type - Visualizer type ('waveform', 'frequency', 'circular')
   */
  setVisualizerType(type) {
    if (['waveform', 'frequency', 'circular'].includes(type)) {
      this.config.visualizerType = type;
    }
  }

  /**
   * Get current audio data
   * @returns {Object} Object with audio metadata
   */
  getAudioData() {
    return {
      currentTime: this.currentTime,
      duration: this.duration,
      isPlaying: this.isPlaying,
      volume: this.audioElement.volume,
      loop: this.audioElement.loop,
      src: this.audioElement.src
    };
  }

  /**
   * Get frequency data
   * @returns {Uint8Array} Frequency data array
   */
  getFrequencyData() {
    if (!this.analyser) return new Uint8Array(0);
    
    const data = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(data);
    return data;
  }

  /**
   * Get waveform data
   * @returns {Uint8Array} Waveform data array
   */
  getWaveformData() {
    if (!this.analyser) return new Uint8Array(0);
    
    const data = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteTimeDomainData(data);
    return data;
  }

  /**
   * Destroy the player and clean up
   */
  destroy() {
    // Stop audio
    this.stop();
    
    // Disconnect audio nodes
    if (this.sourceNode) {
      this.sourceNode.disconnect();
    }
    
    if (this.analyser) {
      this.analyser.disconnect();
    }
    
    // Stop visualization
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
    
    // Close audio context
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
    
    // Remove from global object
    if (window.harmonicVAE) {
      delete window.harmonicVAE.audioPlayer;
      delete window.harmonicVAE.audioAnalyzer;
    }
    
    // Clear HTML
    this.container.innerHTML = '';
  }
}
