/**
 * app.js
 * Main application logic for the HarmonicVAE Feedback System
 * Connects the UI components and handles user interactions
 */

// Define the main application namespace
window.harmonicVAE = window.harmonicVAE || {};

/**
 * Initialize the application when the DOM is fully loaded
 */
document.addEventListener('DOMContentLoaded', () => {
  // Initialize components
  initAudioPlayer();
  initEmotionWheel();
  initStarRating();
  initCommentPrompts();
  initFeedbackForm();
  
  // Load initial audio
  loadDemoAudio('sample1.mp3');
  
  // Register service worker for offline functionality
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/service-worker.js')
      .then(registration => {
        console.log('ServiceWorker registration successful:', registration.scope);
      })
      .catch(error => {
        console.log('ServiceWorker registration failed:', error);
      });
  }
});

/**
 * Initialize the audio player
 */
function initAudioPlayer() {
  const audioPlayer = new HarmonicAudioPlayer({
    containerId: 'audio-player-container',
    visualizerType: 'frequency',
    onPlay: () => {
      document.getElementById('generation-info').classList.add('playing');
      trackPlaybackStart();
    },
    onPause: () => {
      document.getElementById('generation-info').classList.remove('playing');
    },
    onEnded: () => {
      document.getElementById('generation-info').classList.remove('playing');
      showFeedbackPrompt();
    },
    onTimeUpdate: (currentTime, duration) => {
      updateListenProgress(currentTime, duration);
    }
  });
  
  // Store reference in global namespace
  harmonicVAE.player = audioPlayer;
  
  // Add button event listeners
  document.getElementById('generate-button').addEventListener('click', () => {
    generateNewTrack();
  });
}

/**
 * Initialize the emotion wheel
 */
function initEmotionWheel() {
  const emotionWheel = new EmotionWheel({
    containerId: 'emotion-wheel-container',
    onEmotionSelected: (emotion, intensity) => {
      console.log('Emotion selected:', emotion.name, 'Intensity:', intensity);
      updateSelectedEmotion(emotion, intensity);
    },
    onIntensityChange: (emotion, intensity) => {
      console.log('Intensity changed:', intensity);
      updateSelectedEmotion(emotion, intensity);
    },
    showTooltips: true,
    theme: 'colorful',
    animate: true
  });
  
  // Store reference in global namespace
  harmonicVAE.emotionWheel = emotionWheel;
}

/**
 * Initialize the star rating component
 */
function initStarRating() {
  const container = document.getElementById('star-rating-container');
  
  // Create stars
  for (let i = 1; i <= 5; i++) {
    const star = document.createElement('span');
    star.className = 'rating-star';
    star.innerHTML = '★';
    star.dataset.value = i;
    star.addEventListener('click', (e) => {
      selectRating(i);
    });
    star.addEventListener('mouseover', (e) => {
      highlightStars(i);
    });
    container.appendChild(star);
  }
  
  // Add mouseleave event to reset to selected rating
  container.addEventListener('mouseleave', () => {
    const selectedRating = harmonicVAE.selectedRating || 0;
    highlightStars(selectedRating);
  });
  
  // Add helper text
  const helperText = document.createElement('div');
  helperText.className = 'rating-helper-text';
  helperText.textContent = 'Click to rate this composition';
  container.appendChild(helperText);
}

/**
 * Highlight stars up to the specified value
 * @param {number} value - Star rating value to highlight
 */
function highlightStars(value) {
  const stars = document.querySelectorAll('.rating-star');
  stars.forEach((star, index) => {
    if (index < value) {
      star.classList.add('active');
    } else {
      star.classList.remove('active');
    }
  });
  
  // Update helper text based on selection
  const helperTexts = {
    0: 'Click to rate this composition',
    1: 'Poor - Not enjoyable',
    2: 'Fair - Has some merit',
    3: 'Good - Decent composition',
    4: 'Very Good - Above average',
    5: 'Excellent - Outstanding!'
  };
  
  const helperText = document.querySelector('.rating-helper-text');
  if (helperText) {
    helperText.textContent = helperTexts[value] || helperTexts[0];
  }
}

/**
 * Select a rating value
 * @param {number} value - Selected rating value
 */
function selectRating(value) {
  harmonicVAE.selectedRating = value;
  highlightStars(value);
  
  // Update form validation state
  validateFeedbackForm();
}

/**
 * Initialize rotating comment prompts
 */
function initCommentPrompts() {
  const container = document.getElementById('comment-prompt-container');
  const textArea = document.getElementById('comment-input');
  const promptDisplay = document.createElement('div');
  promptDisplay.className = 'prompt-display';
  
  const prompts = [
    "How did this music make you feel emotionally?",
    "Did this composition remind you of any specific memories?",
    "What's the strongest element in this piece?",
    "Where would you imagine hearing this music?",
    "If this music were a color, what would it be and why?",
    "Was there anything surprising or unexpected in this piece?",
    "Does this music tell a story? What might it be?",
    "Would this composition fit in a film or game? Which one?",
    "How would you describe this to someone who hasn't heard it?",
    "What instruments or sounds would make this composition better?"
  ];
  
  // Choose a random prompt
  const randomPrompt = prompts[Math.floor(Math.random() * prompts.length)];
  promptDisplay.textContent = randomPrompt;
  
  // Insert prompt before textarea
  container.insertBefore(promptDisplay, textArea);
  
  // Add refresh button to get new prompt
  const refreshButton = document.createElement('button');
  refreshButton.className = 'refresh-prompt-button';
  refreshButton.innerHTML = '<i class="fas fa-sync-alt"></i>';
  refreshButton.setAttribute('aria-label', 'Get new prompt');
  refreshButton.setAttribute('title', 'Get a different question');
  
  refreshButton.addEventListener('click', () => {
    // Choose a different prompt than the current one
    let newPrompt;
    do {
      newPrompt = prompts[Math.floor(Math.random() * prompts.length)];
    } while (newPrompt === promptDisplay.textContent && prompts.length > 1);
    
    // Animate prompt change
    promptDisplay.style.opacity = 0;
    setTimeout(() => {
      promptDisplay.textContent = newPrompt;
      promptDisplay.style.opacity = 1;
    }, 300);
  });
  
  container.insertBefore(refreshButton, textArea);
  
  // Store current prompt in global namespace
  harmonicVAE.currentPrompt = randomPrompt;
}

/**
 * Initialize the feedback form
 */
function initFeedbackForm() {
  const form = document.getElementById('feedback-form');
  const submitButton = document.getElementById('submit-feedback');
  const consentCheckbox = document.getElementById('consent-checkbox');
  
  // Initialize form validation state
  validateFeedbackForm();
  
  // Add event listeners
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    submitFeedback();
  });
  
  consentCheckbox.addEventListener('change', () => {
    validateFeedbackForm();
  });
  
  document.getElementById('comment-input').addEventListener('input', () => {
    // Optional validation for comments
  });
}

/**
 * Validate the feedback form and update UI accordingly
 */
function validateFeedbackForm() {
  const submitButton = document.getElementById('submit-feedback');
  const consentCheckbox = document.getElementById('consent-checkbox');
  const hasEmotion = !!harmonicVAE.selectedEmotion;
  const hasRating = !!harmonicVAE.selectedRating;
  const hasConsent = consentCheckbox.checked;
  
  // Enable submit button only if all required fields are filled
  submitButton.disabled = !(hasEmotion && hasRating && hasConsent);
  
  // Highlight missing required fields
  document.getElementById('emotion-section').classList.toggle('missing-required', !hasEmotion);
  document.getElementById('rating-section').classList.toggle('missing-required', !hasRating);
  document.getElementById('consent-section').classList.toggle('missing-required', !hasConsent);
  
  return submitButton.disabled === false;
}

/**
 * Load a demo audio file
 * @param {string} filename - Name of the audio file in the demo-audio directory
 */
function loadDemoAudio(filename) {
  const audioPath = `/demo-audio/${filename}`;
  
  // Generate a unique ID for this audio
  const generationId = `demo-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
  
  // Update UI
  const infoElement = document.getElementById('generation-info');
  infoElement.innerHTML = `
    <div class="generation-title">Demo: ${filename}</div>
    <div class="generation-id">ID: ${generationId}</div>
    <div class="generation-time">Generated: ${new Date().toLocaleString()}</div>
  `;
  
  // Show loading indicator
  const loadingIndicator = document.createElement('div');
  loadingIndicator.className = 'loading-indicator';
  loadingIndicator.innerHTML = 'Loading audio...';
  document.getElementById('audio-player-container').appendChild(loadingIndicator);
  
  // Load audio into player
  harmonicVAE.player.loadAudio(audioPath)
    .then(() => {
      // Remove loading indicator
      loadingIndicator.remove();
      
      // Store generation details
      harmonicVAE.currentGeneration = {
        id: generationId,
        filename: filename,
        latentVector: generateMockLatentVector(),
        timestamp: new Date().toISOString()
      };
      
      // Reset feedback form
      resetFeedbackForm();
    })
    .catch(error => {
      console.error('Error loading audio:', error);
      loadingIndicator.innerHTML = 'Error loading audio. Please try again.';
      loadingIndicator.style.color = '#e74c3c';
    });
}

/**
 * Generate a mock latent vector (placeholder for real model output)
 * @returns {Array} Array of latent values
 */
function generateMockLatentVector() {
  const length = 128; // Typical VAE latent dimension
  return Array.from({ length }, () => (Math.random() * 2) - 1);
}

/**
 * Generate a new track (placeholder for real generation)
 */
function generateNewTrack() {
  // In a real implementation, this would call the HarmonicVAE model
  // For demo purposes, we'll just load a different sample
  const samples = ['sample1.mp3', 'sample2.mp3', 'sample3.mp3', 'sample4.mp3', 'sample5.mp3'];
  
  // Get current sample
  const currentFilename = harmonicVAE.currentGeneration?.filename || 'sample1.mp3';
  
  // Choose a different sample
  let newSample;
  do {
    newSample = samples[Math.floor(Math.random() * samples.length)];
  } while (newSample === currentFilename && samples.length > 1);
  
  // Load the new sample
  loadDemoAudio(newSample);
}

/**
 * Track playback start time
 */
function trackPlaybackStart() {
  harmonicVAE.playbackStartTime = Date.now();
}

/**
 * Update listen progress
 * @param {number} currentTime - Current playback time in seconds
 * @param {number} duration - Total duration in seconds
 */
function updateListenProgress(currentTime, duration) {
  // Store listen duration in global namespace
  harmonicVAE.listenDuration = currentTime;
  
  // Update progress indicator if needed
}

/**
 * Show feedback prompt when audio finishes playing
 */
function showFeedbackPrompt() {
  const promptElement = document.getElementById('feedback-prompt');
  
  if (promptElement) {
    promptElement.textContent = 'How was that? Please share your feedback below!';
    promptElement.style.display = 'block';
    
    // Scroll to feedback form
    document.getElementById('feedback-form').scrollIntoView({ 
      behavior: 'smooth',
      block: 'start'
    });
  }
}

/**
 * Update the selected emotion UI
 * @param {Object} emotion - Selected emotion object
 * @param {number} intensity - Emotion intensity (0-1)
 */
function updateSelectedEmotion(emotion, intensity) {
  // Store in global namespace
  harmonicVAE.selectedEmotion = emotion;
  harmonicVAE.emotionIntensity = intensity;
  
  // Update emotion display
  const emotionDisplay = document.getElementById('selected-emotion');
  if (emotionDisplay) {
    emotionDisplay.innerHTML = `
      <div class="emotion-name">${emotion.name}</div>
      <div class="emotion-intensity">Intensity: ${Math.round(intensity * 100)}%</div>
    `;
    
    // Set background color based on emotion
    const emotionColor = getEmotionColor(emotion, intensity);
    emotionDisplay.style.backgroundColor = emotionColor;
    emotionDisplay.style.color = getContrastColor(emotionColor);
    emotionDisplay.style.display = 'block';
  }
  
  // Update form validation
  validateFeedbackForm();
}

/**
 * Get a color representation of the emotion
 * @param {Object} emotion - Emotion object
 * @param {number} intensity - Emotion intensity
 * @returns {string} CSS color
 */
function getEmotionColor(emotion, intensity) {
  // If emotion has color property, use it
  if (emotion.color) {
    // Adjust saturation or lightness based on intensity
    const color = emotion.color;
    
    // If it's hex, convert to HSL, adjust, and convert back
    if (color.startsWith('#')) {
      return adjustColorIntensity(color, intensity);
    }
    
    return color;
  }
  
  // Otherwise use valence and arousal
  if (emotion.valence !== undefined && emotion.arousal !== undefined) {
    // Map valence (0-1) to hue (0-360)
    // Low valence (negative) = blue/purple (240), high valence (positive) = yellow/orange (60)
    const hue = 240 - (emotion.valence * 180);
    
    // Map arousal to saturation
    const saturation = 30 + (emotion.arousal * 50);
    
    // Map intensity to lightness
    const lightness = 40 + (intensity * 30);
    
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  }
  
  // Default color
  return '#6c757d';
}

/**
 * Adjust color intensity
 * @param {string} hexColor - Hex color code
 * @param {number} intensity - Intensity factor (0-1)
 * @returns {string} Adjusted hex color
 */
function adjustColorIntensity(hexColor, intensity) {
  // Simple implementation - blend with white based on intensity
  const r = parseInt(hexColor.slice(1, 3), 16);
  const g = parseInt(hexColor.slice(3, 5), 16);
  const b = parseInt(hexColor.slice(5, 7), 16);
  
  // Adjust based on intensity (higher intensity = more vibrant)
  const factor = 0.5 + (intensity * 0.5);
  
  const adjustedR = Math.min(255, Math.round(r * factor));
  const adjustedG = Math.min(255, Math.round(g * factor));
  const adjustedB = Math.min(255, Math.round(b * factor));
  
  return `#${adjustedR.toString(16).padStart(2, '0')}${adjustedG.toString(16).padStart(2, '0')}${adjustedB.toString(16).padStart(2, '0')}`;
}

/**
 * Get contrast color (black or white) for text
 * @param {string} backgroundColor - Background color
 * @returns {string} Contrast color
 */
function getContrastColor(backgroundColor) {
  // If it's a hex color
  if (backgroundColor.startsWith('#')) {
    const r = parseInt(backgroundColor.slice(1, 3), 16);
    const g = parseInt(backgroundColor.slice(3, 5), 16);
    const b = parseInt(backgroundColor.slice(5, 7), 16);
    
    // Calculate luminance using WCAG formula
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    
    return luminance > 0.5 ? '#000000' : '#ffffff';
  }
  
  // If it's HSL
  if (backgroundColor.startsWith('hsl')) {
    // Extract lightness (simplified)
    const match = backgroundColor.match(/hsl\(\s*\d+\s*,\s*\d+%\s*,\s*(\d+)%\s*\)/);
    if (match && match[1]) {
      const lightness = parseInt(match[1], 10);
      return lightness > 50 ? '#000000' : '#ffffff';
    }
  }
  
  // Default
  return '#ffffff';
}

/**
 * Reset the feedback form
 */
function resetFeedbackForm() {
  // Clear emotion selection
  harmonicVAE.selectedEmotion = null;
  harmonicVAE.emotionIntensity = 0;
  document.getElementById('selected-emotion').style.display = 'none';
  
  // Reset emotion wheel
  if (harmonicVAE.emotionWheel) {
    harmonicVAE.emotionWheel.reset();
  }
  
  // Clear rating
  harmonicVAE.selectedRating = 0;
  highlightStars(0);
  
  // Clear comment
  document.getElementById('comment-input').value = '';
  
  // Reset consent checkbox
  document.getElementById('consent-checkbox').checked = false;
  
  // Hide feedback prompt
  const promptElement = document.getElementById('feedback-prompt');
  if (promptElement) {
    promptElement.style.display = 'none';
  }
  
  // Reset listen duration
  harmonicVAE.listenDuration = 0;
  harmonicVAE.playbackStartTime = null;
  
  // Update form validation
  validateFeedbackForm();
}

/**
 * Submit feedback
 */
function submitFeedback() {
  // Validate form
  if (!validateFeedbackForm()) {
    alert('Please complete all required fields.');
    return;
  }
  
  // Collect feedback data
  const feedbackData = {
    generation_id: harmonicVAE.currentGeneration?.id || 'unknown',
    emotion: {
      name: harmonicVAE.selectedEmotion?.name || 'none',
      valence: harmonicVAE.selectedEmotion?.valence || 0.5,
      arousal: harmonicVAE.selectedEmotion?.arousal || 0.5
    },
    quality_rating: harmonicVAE.selectedRating || 0,
    comments: document.getElementById('comment-input').value,
    prompt_used: harmonicVAE.currentPrompt,
    listen_duration: harmonicVAE.listenDuration || 0,
    consent: document.getElementById('consent-checkbox').checked,
    latent_vector: harmonicVAE.currentGeneration?.latentVector || [],
    condition_params: harmonicVAE.currentGeneration?.conditionParams || {},
    timestamp: new Date().toISOString()
  };
  
  // In a production environment, send to server
  // For demo, we'll log to console and simulate success
  console.log('Feedback data:', feedbackData);
  
  // Show success message
  showFeedbackSuccess();
  
  // Reset form after submission
  setTimeout(() => {
    resetFeedbackForm();
  }, 2000);
}

/**
 * Show feedback success message
 */
function showFeedbackSuccess() {
  // Create success message
  const successMessage = document.createElement('div');
  successMessage.className = 'feedback-success';
  successMessage.innerHTML = `
    <div class="success-icon"><i class="fas fa-check-circle"></i></div>
    <div class="success-text">Thanks for your feedback!</div>
    <div class="success-subtext">Your insights help train HarmonicVAE to create better music.</div>
  `;
  
  // Apply styles
  Object.assign(successMessage.style, {
    position: 'fixed',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    backgroundColor: '#2ecc71',
    color: 'white',
    padding: '20px',
    borderRadius: '10px',
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)',
    textAlign: 'center',
    zIndex: '1000',
    opacity: '0',
    transition: 'opacity 0.3s ease'
  });
  
  // Add to document
  document.body.appendChild(successMessage);
  
  // Animate in
  setTimeout(() => {
    successMessage.style.opacity = '1';
  }, 10);
  
  // Remove after delay
  setTimeout(() => {
    successMessage.style.opacity = '0';
    setTimeout(() => {
      successMessage.remove();
    }, 300);
  }, 2000);
}
