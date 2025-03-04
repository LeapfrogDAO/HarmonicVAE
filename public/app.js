/**
 * app.js
 * Main application logic for the HarmonicVAE Feedback System
 * Adapted to work with existing CSS structure
 */

// Create global namespace
window.harmonicVAE = window.harmonicVAE || {};

// Current feedback data
let currentFeedback = {
  emotion: null,
  rating: 0,
  comment: '',
  consent: false,
  generationId: '',
  listenDuration: 0
};

// Track playback timing
let playbackStartTime = 0;

/**
 * Initialize the application when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', () => {
  // Initialize components
  initAudioPlayer();
  initEmotionWheel();
  initStarRating();
  initFeedbackForm();

  // Load initial demo audio
  loadDemoAudio('sample1.mp3');

  // Initialize UI interactions
  initUIInteractions();
});

/**
 * Initialize the audio player
 */
function initAudioPlayer() {
  const audioPlayer = new HarmonicAudioPlayer({
    containerId: 'audio-player-container',
    visualize: true,
    onPlay: () => {
      // Track playback start time for duration calculation
      playbackStartTime = Date.now();
    },
    onEnded: () => {
      // Calculate listen duration
      if (playbackStartTime > 0) {
        const endTime = Date.now();
        currentFeedback.listenDuration = (endTime - playbackStartTime) / 1000;
        playbackStartTime = 0;
      }

      // Show feedback prompt
      showNotification('Please share your feedback on what you just heard!', 'info');
    }
  });

  // Store in global namespace
  harmonicVAE.audioPlayer = audioPlayer;
}

/**
 * Initialize the emotion wheel
 */
function initEmotionWheel() {
  const emotionWheel = new EmotionWheel({
    containerId: 'emotion-wheel',
    width: 300,
    height: 300,
    onSelectEmotion: (emotion) => {
      // Update current feedback data
      currentFeedback.emotion = emotion;
      
      // Check form validity
      validateForm();
    }
  });

  // Store in global namespace
  harmonicVAE.emotionWheel = emotionWheel;
}

/**
 * Initialize star rating component
 */
function initStarRating() {
  const container = document.getElementById('rating-container');
  if (!container) return;

  // Create SVG element
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', '250');
  svg.setAttribute('height', '50');
  svg.setAttribute('viewBox', '0 0 250 50');
  svg.setAttribute('class', 'star-rating');
  container.appendChild(svg);

  // Create rating display
  const ratingValue = document.createElement('div');
  ratingValue.id = 'rating-value';
  container.appendChild(ratingValue);

  // Create stars
  for (let i = 0; i < 5; i++) {
    const star = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    
    // Draw star shape
    star.setAttribute('d', 'M10,1.2l2.2,6.6H19l-5.4,4l2.1,6.6l-5.4-4l-5.4,4l2.1-6.6L1.6,7.8h6.6L10,1.2z');
    star.setAttribute('transform', `translate(${i * 50 + 25}, 25) scale(2)`);
    star.setAttribute('class', 'star');
    star.setAttribute('data-value', i + 1);
    svg.appendChild(star);

    // Add event listeners
    star.addEventListener('click', (event) => {
      const value = parseInt(event.target.getAttribute('data-value'));
      setRating(value);
    });

    star.addEventListener('mouseover', (event) => {
      const value = parseInt(event.target.getAttribute('data-value'));
      highlightStars(value);
    });
  }

  // Reset stars on mouse leave
  svg.addEventListener('mouseleave', () => {
    highlightStars(currentFeedback.rating);
  });
}

/**
 * Highlight stars up to the specified value
 * @param {number} value - Rating value (1-5)
 */
function highlightStars(value) {
  const stars = document.querySelectorAll('.star');
  
  stars.forEach((star, index) => {
    if (index < value) {
      star.classList.add('selected');
    } else {
      star.classList.remove('selected');
    }
  });

  // Update rating display text
  const ratingTexts = [
    '',
    'Poor',
    'Fair',
    'Good',
    'Very Good',
    'Excellent'
  ];

  const ratingDisplay = document.getElementById('rating-value');
  if (ratingDisplay) {
    ratingDisplay.textContent = value > 0 ? ratingTexts[value] : '';
  }
}

/**
 * Set rating value
 * @param {number} value - Rating value (1-5)
 */
function setRating(value) {
  currentFeedback.rating = value;
  highlightStars(value);
  validateForm();
}

/**
 * Initialize the feedback form
 */
function initFeedbackForm() {
  const commentInput = document.getElementById('comment-input');
  const consentCheckbox = document.getElementById('consent-checkbox');
  const submitButton = document.getElementById('submit-button');

  if (commentInput) {
    commentInput.addEventListener('input', () => {
      currentFeedback.comment = commentInput.value;
    });
  }

  if (consentCheckbox) {
    consentCheckbox.addEventListener('change', () => {
      currentFeedback.consent = consentCheckbox.checked;
      validateForm();
    });
  }

  if (submitButton) {
    submitButton.addEventListener('click', (event) => {
      event.preventDefault();
      submitFeedback();
    });
    submitButton.disabled = true;
  }
}

/**
 * Initialize UI interactions (modals, etc.)
 */
function initUIInteractions() {
  // New track button
  const generateButton = document.getElementById('generate-button');
  if (generateButton) {
    generateButton.addEventListener('click', () => {
      generateNewTrack();
    });
  }

  // Reset feedback button if available
  const resetButton = document.getElementById('reset-button');
  if (resetButton) {
    resetButton.addEventListener('click', () => {
      resetFeedbackForm();
    });
  }

  // Handle any modals
  const modalButtons = document.querySelectorAll('[data-modal]');
  
  modalButtons.forEach(button => {
    button.addEventListener('click', () => {
      const modalId = button.getAttribute('data-modal');
      const modal = document.getElementById(modalId);
      
      if (modal) {
        modal.classList.add('visible');
      }
    });
  });

  // Close modals when clicking close buttons
  const closeButtons = document.querySelectorAll('.close-modal');
  closeButtons.forEach(button => {
    button.addEventListener('click', () => {
      const modal = button.closest('.modal');
      if (modal) {
        modal.classList.remove('visible');
      }
    });
  });
}

/**
 * Load a demo audio track
 * @param {string} filename - Audio filename
 */
function loadDemoAudio(filename) {
  const audioPath = `/demo-audio/${filename}`;
  
  // Generate a unique ID for this session
  const generationId = `demo-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
  currentFeedback.generationId = generationId;
  
  // Show loading state
  showNotification('Loading audio...', 'info');
  
  // Update track info if element exists
  const trackInfo = document.getElementById('track-info');
  if (trackInfo) {
    trackInfo.innerHTML = `
      <div><strong>Track:</strong> ${filename}</div>
      <div><strong>ID:</strong> ${generationId}</div>
      <div><strong>Generated:</strong> ${new Date().toLocaleString()}</div>
    `;
  }
  
  // Load audio into player
  if (harmonicVAE.audioPlayer) {
    harmonicVAE.audioPlayer.loadAudio(audioPath)
      .then(() => {
        showNotification('Audio loaded successfully!', 'success');
        resetFeedbackForm();
      })
      .catch(error => {
        console.error('Error loading audio:', error);
        showNotification('Failed to load audio. Please try again.', 'error');
      });
  }
}
