/**
 * emotion-wheel.js
 * Emotion wheel component for HarmonicVAE Feedback System
 * Adapted to work with existing CSS structure
 */

class EmotionWheel {
  /**
   * Constructor for the EmotionWheel component
   * @param {Object} config - Configuration object
   */
  constructor(config) {
    this.config = {
      containerId: 'emotion-wheel',
      width: 300,
      height: 300,
      onSelectEmotion: (emotion) => {},
      ...config
    };

    this.selectedEmotion = null;
    this.init();
  }

  /**
   * Initialize the wheel component
   */
  init() {
    this.container = document.getElementById(this.config.containerId);
    if (!this.container) {
      console.error(`Container with ID '${this.config.containerId}' not found`);
      return;
    }

    // Clear container
    this.container.innerHTML = '';

    // Create SVG element
    this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.svg.setAttribute('width', this.config.width);
    this.svg.setAttribute('height', this.config.height);
    this.svg.setAttribute('class', 'emotion-wheel');
    this.svg.setAttribute('viewBox', `0 0 ${this.config.width} ${this.config.height}`);
    this.container.appendChild(this.svg);

    // Create selection display
    this.selectionDisplay = document.createElement('div');
    this.selectionDisplay.className = 'selection-display';
    this.container.appendChild(this.selectionDisplay);

    // Draw the emotion wheel
    this.drawWheel();

    // Add event listeners
    window.addEventListener('resize', this.handleResize.bind(this));
  }

  /**
   * Draw the emotion wheel
   */
  drawWheel() {
    const centerX = this.config.width / 2;
    const centerY = this.config.height / 2;
    const radius = Math.min(centerX, centerY) - 10;

    // Define emotions
    const emotions = [
      { id: 'joy', name: 'Joy', color: '#FFD700', textColor: '#000000', position: 0 },
      { id: 'excitement', name: 'Excitement', color: '#FF8C00', textColor: '#000000', position: 1 },
      { id: 'surprise', name: 'Surprise', color: '#9370DB', textColor: '#FFFFFF', position: 2 },
      { id: 'fear', name: 'Fear', color: '#800080', textColor: '#FFFFFF', position: 3 },
      { id: 'anger', name: 'Anger', color: '#B22222', textColor: '#FFFFFF', position: 4 },
      { id: 'sadness', name: 'Sadness', color: '#4682B4', textColor: '#FFFFFF', position: 5 },
      { id: 'calm', name: 'Calm', color: '#20B2AA', textColor: '#000000', position: 6 },
      { id: 'love', name: 'Love', color: '#FF69B4', textColor: '#000000', position: 7 }
    ];

    // Draw segments
    const segmentAngle = (2 * Math.PI) / emotions.length;

    emotions.forEach((emotion, index) => {
      const startAngle = index * segmentAngle;
      const endAngle = (index + 1) * segmentAngle;

      // Calculate segment path
      const x1 = centerX + radius * Math.cos(startAngle);
      const y1 = centerY + radius * Math.sin(startAngle);
      const x2 = centerX + radius * Math.cos(endAngle);
      const y2 = centerY + radius * Math.sin(endAngle);

      const largeArcFlag = endAngle - startAngle > Math.PI ? 1 : 0;

      const pathData = [
        `M ${centerX} ${centerY}`,
        `L ${x1} ${y1}`,
        `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}`,
        'Z'
      ].join(' ');

      // Create segment
      const segment = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      segment.setAttribute('d', pathData);
      segment.setAttribute('fill', emotion.color);
      segment.setAttribute('class', 'emotion-circle');
      segment.setAttribute('data-emotion-id', emotion.id);
      segment.setAttribute('data-emotion-name', emotion.name);
      this.svg.appendChild(segment);

      // Add event listeners
      segment.addEventListener('click', () => this.selectEmotion(emotion, segment));

      // Add label
      const labelAngle = startAngle + segmentAngle / 2;
      const labelRadius = radius * 0.7;
      const labelX = centerX + labelRadius * Math.cos(labelAngle);
      const labelY = centerY + labelRadius * Math.sin(labelAngle);

      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', labelX);
      text.setAttribute('y', labelY);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('dominant-baseline', 'middle');
      text.setAttribute('fill', emotion.textColor);
      text.setAttribute('font-size', '14px');
      text.setAttribute('font-weight', 'bold');
      text.setAttribute('pointer-events', 'none');
      text.textContent = emotion.name;

      // Add rotation to text for better readability
      const rotation = (labelAngle * 180 / Math.PI) - 90;
      const adjustedRotation = (rotation > 90 && rotation < 270) ? rotation + 180 : rotation;
      text.setAttribute('transform', `rotate(${adjustedRotation}, ${labelX}, ${labelY})`);

      this.svg.appendChild(text);
    });

    // Draw center circle
    const centerCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    centerCircle.setAttribute('cx', centerX);
    centerCircle.setAttribute('cy', centerY);
    centerCircle.setAttribute('r', radius * 0.15);
    centerCircle.setAttribute('fill', '#FFFFFF');
    centerCircle.setAttribute('class', 'emotion-circle');
    centerCircle.setAttribute('data-emotion-id', 'neutral');
    centerCircle.setAttribute('data-emotion-name', 'Neutral');
    centerCircle.addEventListener('click', () => {
      this.selectEmotion({ id: 'neutral', name: 'Neutral' }, centerCircle);
    });
    this.svg.appendChild(centerCircle);

    // Add center text
    const centerText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    centerText.setAttribute('x', centerX);
    centerText.setAttribute('y', centerY);
    centerText.setAttribute('text-anchor', 'middle');
    centerText.setAttribute('dominant-baseline', 'middle');
    centerText.setAttribute('fill', '#333333');
    centerText.setAttribute('font-size', '14px');
    centerText.setAttribute('pointer-events', 'none');
    centerText.textContent = 'Neutral';
    this.svg.appendChild(centerText);
  }

  /**
   * Handle selection of an emotion
   * @param {Object} emotion - The selected emotion
   * @param {Element} segment - The selected segment
   */
  selectEmotion(emotion, segment) {
    // Clear previous selection
    const allSegments = this.svg.querySelectorAll('.emotion-circle');
    allSegments.forEach(s => s.classList.remove('selected'));

    // Highlight selected segment
    segment.classList.add('selected');

    // Update selection display
    this.selectionDisplay.textContent = `Selected: ${emotion.name}`;

    // Store selected emotion
    this.selectedEmotion = emotion;

    // Call callback
    this.config.onSelectEmotion(emotion);
  }

  /**
   * Get the currently selected emotion
   * @returns {Object|null} The selected emotion or null
   */
  getSelectedEmotion() {
    return this.selectedEmotion;
  }

  /**
   * Handle resize events
   */
  handleResize() {
    const containerWidth = this.container.clientWidth;
    const containerHeight = this.container.clientHeight;
    
    // Only resize if container dimensions changed significantly
    if (Math.abs(containerWidth - this.config.width) > 50 ||
        Math.abs(containerHeight - this.config.height) > 50) {
      this.config.width = containerWidth;
      this.config.height = containerHeight;
      
      // Redraw wheel
      this.svg.innerHTML = '';
      this.svg.setAttribute('width', this.config.width);
      this.svg.setAttribute('height', this.config.height);
      this.svg.setAttribute('viewBox', `0 0 ${this.config.width} ${this.config.height}`);
      this.drawWheel();
    }
  }

  /**
   * Reset the selection
   */
  reset() {
    const allSegments = this.svg.querySelectorAll('.emotion-circle');
    allSegments.forEach(s => s.classList.remove('selected'));
    this.selectionDisplay.textContent = '';
    this.selectedEmotion = null;
  }
}
