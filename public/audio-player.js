/**
 * audio-player.js
 * Enhanced audio player for HarmonicVAE Feedback System
 * Adapted to work with existing CSS structure
 */

class HarmonicAudioPlayer {
  /**
   * Constructor for the HarmonicAudioPlayer
   * @param {Object} config - Configuration object
   */
  constructor(config) {
    this.config = {
      containerId: 'audio-player-container',
      audioPath: null,
      visualize: true,
      onPlay: () => {},
      onPause: () => {},
      onEnded: () => {},
      onTimeUpdate: () => {},
      ...config
    };

    this.isPlaying = false;
    this.currentTime = 0;
    this.duration = 0;
    this.audioContext = null;
    this.analyser = null;
    this.dataArray = null;
    this.visualizationActive = false;

    this.init();
  }

  /**
   * Initialize the audio player
   */
  init() {
    this.container = document.getElementById(this.config.containerId);
    if (!this.container) {
      console.error(`Container with ID '${this.config.containerId}' not found`);
      return;
    }

    // Create audio context if Web Audio API is supported
    try {
      window.AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioContext();
    } catch (e) {
      console.warn('Web Audio API is not supported in this browser. Visualization will be disabled.');
      this.config.visualize = false;
    }

    // Create HTML structure
    this.createPlayerElements();

    // Set up audio analyzer
    if (this.config.visualize && this.audioContext) {
      this.setupAudioAnalyzer();
    }

    // Load initial audio if provided
    if (this.config.audioPath) {
      this.loadAudio(this.config.audioPath);
    }

    // Set up global reference for other components
    if (!window.harmonicVAE) {
      window.harmonicVAE = {};
    }
    window.harmonicVAE.audioPlayer = this;
  }

  /**
   * Create player HTML elements
   */
  createPlayerElements() {
    // Clear container
    this.container.innerHTML = '';
    
    // Create player wrapper
    const playerWrapper = document.createElement('div');
    playerWrapper.className = 'audio-player';
    
    // Create audio element
    this.audioElement = document.createElement('audio');
    this.audioElement.className = 'audio-element';
    this.audioElement.setAttribute('controls', '');
    this.audioElement.setAttribute('preload', 'auto');
    
    // Create visualizer canvas if enabled
    if (this.config.visualize) {
      this.visualizerContainer = document.createElement('div');
      this.visualizerContainer.className = 'visualizer-container';
      
      this.canvas = document.createElement('canvas');
      this.canvas.className = 'visualizer';
      this.canvas.width = this.container.clientWidth;
      this.canvas.height = 80;
      
      this.visualizerContainer.appendChild(this.canvas);
      this.canvasContext = this.canvas.getContext('2d');
    }
    
    // Add generation button
    this.generateButton = document.createElement('button');
    this.generateButton.className = 'primary-button';
    this.generateButton.textContent = 'Generate New Track';
    this.generateButton.addEventListener('click', () => {
      // Call external function if it exists
      if (typeof generateNewTrack === 'function') {
        generateNewTrack();
      } else {
        this.dispatchEvent('generate');
      }
    });
    
    // Append elements to wrapper
    playerWrapper.appendChild(this.audioElement);
    if (this.config.visualize) {
      playerWrapper.appendChild(this.visualizerContainer);
    }
    playerWrapper.appendChild(this.generateButton);
    
    // Append wrapper to container
    this.container.appendChild(playerWrapper);
    
    // Add event listeners to audio element
    this.addAudioEventListeners();
  }

  /**
   * Add event listeners to the audio element
   */
  addAudioEventListeners() {
    this.audioElement.addEventListener('play', () => {
      this.isPlaying = true;
      if (this.audioContext && this.audioContext.state === 'suspended') {
        this.audioContext.resume();
      }
      this.startVisualization();
      this.config.onPlay();
      this.dispatchEvent('play');
    });
    
    this.audioElement.addEventListener('pause', () => {
      this.isPlaying = false;
      this.stopVisualization();
      this.config.onPause();
      this.dispatchEvent('pause');
    });
    
    this.audioElement.addEventListener('ended', () => {
      this.isPlaying = false;
      this.stopVisualization();
      this.config.onEnded();
      this.dispatchEvent('ended');
    });
    
    this.audioElement.addEventListener('timeupdate', () => {
      this.currentTime = this.audioElement.currentTime;
      this.config.onTimeUpdate(this.currentTime, this.duration);
      this.dispatchEvent('timeupdate', { currentTime: this.currentTime, duration: this.duration });
    });
    
    this.audioElement.addEventListener('loadedmetadata', () => {
      this.duration = this.audioElement.duration;
      this.dispatchEvent('loaded', { duration: this.duration });
    });
  }

  /**
   * Set up audio analyzer for visualization
   */
  setupAudioAnalyzer() {
    if (!this.audioContext) return;
    
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 256;
    this.bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(this.bufferLength);
  }

  /**
   * Connect audio element to analyzer
   */
  connectAudioSource() {
    if (!this.audioContext || !this.analyser) return;
    
    // Create media element source if not already created
    if (!this.source) {
      this.source = this.audioContext.createMediaElementSource(this.audioElement);
      this.source.connect(this.analyser);
      this.analyser.connect(this.audioContext.destination);
    }
  }

  /**
   * Start visualization animation
   */
  startVisualization() {
    if (!this.config.visualize || !this.canvasContext || this.visualizationActive) return;
    
    this.connectAudioSource();
    this.visualizationActive = true;
    this.drawVisualization();
  }

  /**
   * Stop visualization animation
   */
  stopVisualization() {
    this.visualizationActive = false;
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  /**
   * Draw visualization frame
   */
  drawVisualization() {
    if (!this.visualizationActive) return;
    
    this.animationFrame = requestAnimationFrame(() => this.drawVisualization());
    
    this.analyser.getByteFrequencyData(this.dataArray);
    
    const width = this.canvas.width;
    const height = this.canvas.height;
    const barWidth = (width / this.bufferLength) * 2.5;
    
    this.canvasContext.clearRect(0, 0, width, height);
    this.canvasContext.fillStyle = 'rgb(0, 0, 0, 0.1)';
    this.canvasContext.fillRect(0, 0, width, height);
    
    let x = 0;
    
    for (let i = 0; i < this.bufferLength; i++) {
      const barHeight = this.dataArray[i] / 255 * height;
      
      // Use CSS variable colors for visualization
      const hue = (i / this.bufferLength) * 360;
      this.canvasContext.fillStyle = `hsl(${hue}, 70%, 60%)`;
      this.canvasContext.fillRect(x, height - barHeight, barWidth, barHeight);
      
      x += barWidth + 1;
      if (x > width) break;
    }
  }

  /**
   * Load audio from URL
   * @param {string} url - URL of the audio file
   * @returns {Promise} Promise that resolves when audio is loaded
   */
  loadAudio(url) {
    return new Promise((resolve, reject) => {
      this.audioElement.src = url;
      this.audioElement.load();
      
      const loadHandler = () => {
        resolve();
        this.audioElement.removeEventListener('canplaythrough', loadHandler);
      };
      
      const errorHandler = (err) => {
        reject(err);
        this.audioElement.removeEventListener('error', errorHandler);
      };
      
      this.audioElement.addEventListener('canplaythrough', loadHandler, { once: true });
      this.audioElement.addEventListener('error', errorHandler, { once: true });
    });
  }

  /**
   * Play audio
   */
  play() {
    if (this.audioElement.readyState >= 2) {
      this.audioElement.play();
    }
  }

  /**
   * Pause audio
   */
  pause() {
    this.audioElement.pause();
  }

  /**
   * Toggle play/pause
   */
  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  /**
   * Seek to specific time
   * @param {number} time - Time in seconds
   */
  seek(time) {
    if (this.audioElement.readyState >= 2) {
      this.audioElement.currentTime = Math.max(0, Math.min(time, this.duration));
    }
  }

  /**
   * Set volume
   * @param {number} volume - Volume level (0-1)
   */
  setVolume(volume) {
    this.audioElement.volume = Math.max(0, Math.min(1, volume));
  }

  /**
   * Get current audio state
   * @returns {Object} Current audio state
   */
  getState() {
    return {
      isPlaying: this.isPlaying,
      currentTime: this.currentTime,
      duration: this.duration,
      volume: this.audioElement.volume
    };
  }

  /**
   * Dispatch custom event
   * @param {string} name - Event name
   * @param {Object} detail - Event details
   */
  dispatchEvent(name, detail = {}) {
    const event = new CustomEvent(`harmonic-player:${name}`, {
      detail: { ...detail, player: this },
      bubbles: true
    });
    this.container.dispatchEvent(event);
  }

  /**
   * Get frequency data for external visualization
   * @returns {Uint8Array} Frequency data
   */
  getFrequencyData() {
    if (!this.analyser || !this.dataArray) return new Uint8Array();
    
    this.analyser.getByteFrequencyData(this.dataArray);
    return this.dataArray;
  }

  /**
   * Clean up resources
   */
  destroy() {
    this.stopVisualization();
    
    if (this.source) {
      this.source.disconnect();
    }
    
    if (this.analyser) {
      this.analyser.disconnect();
    }
    
    if (this.audioContext) {
      this.audioContext.close();
    }
    
    this.container.innerHTML = '';
    
    if (window.harmonicVAE && window.harmonicVAE.audioPlayer === this) {
      delete window.harmonicVAE.audioPlayer;
    }
  }
}
