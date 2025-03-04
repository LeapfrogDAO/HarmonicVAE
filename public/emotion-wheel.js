/**
 * emotion-wheel.js
 * A comprehensive, interactive emotion wheel for the HarmonicVAE Feedback System
 * Uses D3.js for SVG manipulation and animations
 * 
 * Features:
 * - Hierarchical emotion representation (primary, secondary, tertiary emotions)
 * - Smooth animations and transitions
 * - Responsive design
 * - Accessibility support
 * - Tooltips and information display
 * - Event handling and callbacks
 * - Emotion intensity tracking (distance from center)
 * - Color mapping based on emotion valence and arousal
 */

class EmotionWheel {
  /**
   * Constructor for the EmotionWheel component
   * @param {Object} config - Configuration object
   * @param {string} config.containerId - ID of the container element
   * @param {number} config.width - Width of the wheel (default: container width)
   * @param {number} config.height - Height of the wheel (default: container height)
   * @param {number} config.margin - Margin around the wheel (default: 50)
   * @param {boolean} config.responsive - Whether the wheel should be responsive (default: true)
   * @param {Object} config.emotions - Emotion data structure (default: standard emotion wheel)
   * @param {Function} config.onEmotionSelected - Callback when emotion is selected
   * @param {Function} config.onIntensityChange - Callback when emotion intensity changes
   * @param {boolean} config.showLabels - Whether to show all emotion labels (default: true)
   * @param {boolean} config.showTooltips - Whether to show tooltips (default: true)
   * @param {string} config.theme - Color theme ('light', 'dark', 'colorful') (default: 'colorful')
   * @param {boolean} config.animate - Whether to use animations (default: true)
   * @param {number} config.animationDuration - Duration of animations in ms (default: 500)
   */
  constructor(config) {
    // Default configuration
    this.config = {
      containerId: 'emotion-wheel-container',
      width: null,
      height: null,
      margin: 50,
      responsive: true,
      showLabels: true,
      showTooltips: true,
      theme: 'colorful',
      animate: true,
      animationDuration: 500,
      onEmotionSelected: () => {},
      onIntensityChange: () => {},
      emotions: null,
      ...config
    };

    // State
    this.state = {
      selectedEmotion: null,
      selectedIntensity: 0,
      currentLevel: 0, // 0 = primary, 1 = secondary, 2 = tertiary
      hoveredEmotion: null,
      wheelRotation: 0,
      zoomedEmotion: null,
      isTransitioning: false
    };

    // Default emotion structure if not provided
    if (!this.config.emotions) {
      this.config.emotions = this._defaultEmotionStructure();
    }

    // Initialize
    this._init();
  }

  /**
   * Initialize the emotion wheel
   * @private
   */
  _init() {
    // Get container and set dimensions
    this.container = document.getElementById(this.config.containerId);
    if (!this.container) {
      console.error(`Container with ID '${this.config.containerId}' not found`);
      return;
    }

    // Set up dimensions
    const containerRect = this.container.getBoundingClientRect();
    this.config.width = this.config.width || containerRect.width;
    this.config.height = this.config.height || containerRect.height;
    
    // Ensure container has position relative for tooltip positioning
    if (window.getComputedStyle(this.container).position === 'static') {
      this.container.style.position = 'relative';
    }

    // Clear any existing content
    this.container.innerHTML = '';

    // Set up SVG
    this.svg = d3.select(this.container)
      .append('svg')
      .attr('width', this.config.width)
      .attr('height', this.config.height)
      .attr('class', 'harmonic-emotion-wheel')
      .attr('aria-label', 'Emotion Wheel for Music Feedback');
    
    // Add description for accessibility
    this.svg.append('desc')
      .text('An interactive wheel to select emotions in response to music. Navigate clockwise to explore from Joy to Sadness to Fear to Anger and back to Joy.');

    // Create main group and center it
    this.mainGroup = this.svg.append('g')
      .attr('transform', `translate(${this.config.width / 2}, ${this.config.height / 2})`);
    
    // Create tooltip
    if (this.config.showTooltips) {
      this.tooltip = d3.select(this.container)
        .append('div')
        .attr('class', 'emotion-wheel-tooltip')
        .style('position', 'absolute')
        .style('visibility', 'hidden')
        .style('background-color', 'rgba(0, 0, 0, 0.8)')
        .style('color', 'white')
        .style('padding', '8px')
        .style('border-radius', '4px')
        .style('pointer-events', 'none')
        .style('z-index', '1000')
        .style('transition', 'opacity 0.2s');
    }

    // Set up event handlers
    this._setupEventHandlers();

    // Determine the wheel radius
    const minDimension = Math.min(this.config.width, this.config.height) - (this.config.margin * 2);
    this.radius = minDimension / 2;

    // Draw the wheel
    this._drawWheel();

    // Render the wheel
    this._renderWheel();
    
    // Add audio reactivity if a reference to audio analyzer is provided
    if (window.harmonicVAE && window.harmonicVAE.audioAnalyzer) {
      this._setupAudioReactivity();
    }

    // Make responsive if needed
    if (this.config.responsive) {
      this._makeResponsive();
    }
  }

  /**
   * Draw the base wheel structure
   * @private
   */
  _drawWheel() {
    // Create the background circle
    this.mainGroup.append('circle')
      .attr('class', 'emotion-wheel-background')
      .attr('r', this.radius)
      .attr('fill', this._getThemeColors().background)
      .attr('stroke', this._getThemeColors().border)
      .attr('stroke-width', 2);

    // Create the center circle (neutral emotion)
    this.mainGroup.append('circle')
      .attr('class', 'emotion-wheel-center')
      .attr('r', this.radius * 0.15)
      .attr('fill', this._getThemeColors().neutral)
      .attr('stroke', this._getThemeColors().border)
      .attr('stroke-width', 1)
      .attr('cursor', 'pointer')
      .on('click', () => this._onCenterClick());

    // Add neutral text
    this.mainGroup.append('text')
      .attr('class', 'emotion-wheel-center-text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', this._getContrastColor(this._getThemeColors().neutral))
      .attr('pointer-events', 'none')
      .attr('font-weight', 'bold')
      .text('Neutral');
  }

  /**
   * Render the emotion wheel segments based on current state
   * @private
   */
  _renderWheel() {
    const self = this;
    const emotionsToRender = this._getEmotionsForCurrentLevel();
    const segmentAngle = (2 * Math.PI) / emotionsToRender.length;
    
    // Remove existing segments
    this.mainGroup.selectAll('.emotion-segment').remove();
    this.mainGroup.selectAll('.emotion-label').remove();
    
    // Inner and outer radius based on current level
    const levelMultiplier = 1 / (this.state.currentLevel + 2);
    const innerRadius = this.radius * 0.15 + (this.state.currentLevel * this.radius * 0.15);
    const outerRadius = this.radius;
    
    // Create an arc generator
    const arc = d3.arc()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius)
      .startAngle((d, i) => i * segmentAngle + this.state.wheelRotation)
      .endAngle((d, i) => (i + 1) * segmentAngle + this.state.wheelRotation);
    
    // Create segments
    const segments = this.mainGroup.selectAll('.emotion-segment')
      .data(emotionsToRender)
      .enter()
      .append('path')
      .attr('class', 'emotion-segment')
      .attr('d', arc)
      .attr('fill', (d, i) => this._getEmotionColor(d, i, emotionsToRender.length))
      .attr('stroke', this._getThemeColors().border)
      .attr('stroke-width', 1)
      .attr('cursor', 'pointer')
      .attr('data-emotion-name', d => d.name)
      .attr('data-emotion-id', d => d.id)
      .attr('aria-label', d => d.name)
      .attr('tabindex', d => 0) // Make focusable
      .on('click', function(event, d) {
        self._onSegmentClick(d, this);
      })
      .on('mouseover', function(event, d) {
        self._onSegmentHover(d, this, event);
      })
      .on('mouseout', function() {
        self._onSegmentLeave();
      })
      .on('mousemove', function(event) {
        if (self.tooltip) {
          const [x, y] = d3.pointer(event, self.container);
          self.tooltip
            .style('left', `${x + 15}px`)
            .style('top', `${y}px`);
        }
      });
    
    // Add animation if enabled
    if (this.config.animate) {
      segments
        .style('opacity', 0)
        .transition()
        .duration(this.config.animationDuration)
        .style('opacity', 1);
    }
    
    // Add labels if enabled
    if (this.config.showLabels) {
      const labels = this.mainGroup.selectAll('.emotion-label')
        .data(emotionsToRender)
        .enter()
        .append('text')
        .attr('class', 'emotion-label')
        .attr('pointer-events', 'none')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', this._getLabelFontSize(emotionsToRender.length))
        .attr('font-weight', 'bold')
        .attr('fill', (d, i) => this._getContrastColor(this._getEmotionColor(d, i, emotionsToRender.length)))
        .attr('transform', (d, i) => {
          const angle = i * segmentAngle + segmentAngle / 2 + this.state.wheelRotation;
          const labelRadius = innerRadius + (outerRadius - innerRadius) * 0.65;
          const x = Math.sin(angle) * labelRadius;
          const y = -Math.cos(angle) * labelRadius;
          const rotation = (angle * 180 / Math.PI) - 90;
          const adjustedRotation = rotation > 90 && rotation < 270 ? rotation + 180 : rotation;
          return `translate(${x}, ${y}) rotate(${adjustedRotation})`;
        })
        .text(d => d.name);
      
      // Add animation if enabled
      if (this.config.animate) {
        labels
          .style('opacity', 0)
          .transition()
          .duration(this.config.animationDuration)
          .delay(this.config.animationDuration * 0.5)
          .style('opacity', 1);
      }
    }
    
    // Add navigation button for going back if not at primary level
    if (this.state.currentLevel > 0) {
      this._addBackButton();
    }
    
    // Update or add the intensity indicator if an emotion is selected
    if (this.state.selectedEmotion) {
      this._updateIntensityIndicator();
    }
  }

  /**
   * Add a back button to return to previous level
   * @private
   */
  _addBackButton() {
    const backButton = this.mainGroup.append('g')
      .attr('class', 'emotion-back-button')
      .attr('cursor', 'pointer')
      .attr('transform', `translate(0, ${-this.radius - 20})`)
      .on('click', () => this._navigateBack());
    
    backButton.append('circle')
      .attr('r', 15)
      .attr('fill', this._getThemeColors().buttonBackground)
      .attr('stroke', this._getThemeColors().border);
    
    backButton.append('path')
      .attr('d', 'M-5,0 L5,10 L5,-10 Z')
      .attr('fill', this._getThemeColors().buttonText)
      .attr('transform', 'rotate(180)');
    
    // Add animation if enabled
    if (this.config.animate) {
      backButton
        .style('opacity', 0)
        .transition()
        .duration(this.config.animationDuration)
        .style('opacity', 1);
    }
  }

  /**
   * Update or create the intensity indicator
   * @private
   */
  _updateIntensityIndicator() {
    // Remove existing indicator
    this.mainGroup.selectAll('.intensity-indicator').remove();
    
    // Get the selected emotion angle
    const emotionsAtCurrentLevel = this._getEmotionsForCurrentLevel();
    const selectedIndex = emotionsAtCurrentLevel.findIndex(e => e.id === this.state.selectedEmotion.id);
    
    if (selectedIndex === -1) return;
    
    const segmentAngle = (2 * Math.PI) / emotionsAtCurrentLevel.length;
    const angle = selectedIndex * segmentAngle + segmentAngle / 2 + this.state.wheelRotation;
    
    // Create the intensity line
    const intensityGroup = this.mainGroup.append('g')
      .attr('class', 'intensity-indicator')
      .attr('transform', `rotate(${angle * 180 / Math.PI})`);
    
    // Inner and outer radius for reference
    const innerRadius = this.radius * 0.15 + (this.state.currentLevel * this.radius * 0.15);
    const outerRadius = this.radius;
    
    // Calculate position based on intensity
    const intensityRadius = innerRadius + (outerRadius - innerRadius) * this.state.selectedIntensity;
    
    // Add the indicator line
    intensityGroup.append('line')
      .attr('x1', 0)
      .attr('y1', -innerRadius)
      .attr('x2', 0)
      .attr('y2', -outerRadius)
      .attr('stroke', this._getThemeColors().intensityLine)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4,4');
    
    // Add the draggable handle
    const handle = intensityGroup.append('circle')
      .attr('class', 'intensity-handle')
      .attr('cx', 0)
      .attr('cy', -intensityRadius)
      .attr('r', 8)
      .attr('fill', this._getThemeColors().handleFill)
      .attr('stroke', this._getThemeColors().handleStroke)
      .attr('stroke-width', 2)
      .attr('cursor', 'ns-resize')
      .call(d3.drag()
        .on('drag', event => this._onIntensityDrag(event, innerRadius, outerRadius))
        .on('end', () => this._onIntensityDragEnd()));
    
    // Add animation if enabled
    if (this.config.animate) {
      handle
        .style('opacity', 0)
        .transition()
        .duration(this.config.animationDuration)
        .style('opacity', 1);
    }
  }

  /**
   * Handle intensity drag event
   * @param {Object} event - Drag event
   * @param {number} innerRadius - Inner radius of the wheel
   * @param {number} outerRadius - Outer radius of the wheel
   * @private
   */
  _onIntensityDrag(event, innerRadius, outerRadius) {
    // Get the y-coordinate of the drag
    const y = Math.max(Math.min(-innerRadius, -event.y), -outerRadius);
    
    // Update the handle position
    d3.select(event.sourceEvent.target)
      .attr('cy', y);
    
    // Calculate the intensity (0-1)
    const intensity = Math.abs((y + innerRadius) / (outerRadius - innerRadius));
    this.state.selectedIntensity = intensity;
    
    // Call the intensity change callback
    this.config.onIntensityChange(this.state.selectedEmotion, intensity);
  }

  /**
   * Handle the end of an intensity drag
   * @private
   */
  _onIntensityDragEnd() {
    // Optional animation or feedback
  }

  /**
   * Navigate back to the previous level
   * @private
   */
  _navigateBack() {
    if (this.state.currentLevel > 0) {
      this.state.isTransitioning = true;
      
      if (this.config.animate) {
        // Animate out current level
        this.mainGroup.selectAll('.emotion-segment, .emotion-label, .emotion-back-button')
          .transition()
          .duration(this.config.animationDuration / 2)
          .style('opacity', 0)
          .on('end', (d, i, nodes) => {
            // Only execute once when the first element's transition ends
            if (i === 0) {
              this.state.currentLevel -= 1;
              this.state.wheelRotation = 0;
              this._renderWheel();
              this.state.isTransitioning = false;
            }
          });
      } else {
        this.state.currentLevel -= 1;
        this.state.wheelRotation = 0;
        this._renderWheel();
        this.state.isTransitioning = false;
      }
    }
  }

  /**
   * Handle segment click event
   * @param {Object} emotion - The clicked emotion
   * @param {Element} element - The clicked element
   * @private
   */
  _onSegmentClick(emotion, element) {
    if (this.state.isTransitioning) return;
    
    // Clone emotion to avoid mutations
    const selectedEmotion = { ...emotion };
    
    // If the emotion has children and we're not at the tertiary level, go deeper
    if (selectedEmotion.children && selectedEmotion.children.length > 0 && this.state.currentLevel < 2) {
      this.state.isTransitioning = true;
      this.state.zoomedEmotion = selectedEmotion;
      
      if (this.config.animate) {
        // Get the index and angle for the selected segment
        const emotionsAtCurrentLevel = this._getEmotionsForCurrentLevel();
        const selectedIndex = emotionsAtCurrentLevel.findIndex(e => e.id === selectedEmotion.id);
        const segmentAngle = (2 * Math.PI) / emotionsAtCurrentLevel.length;
        const selectedAngle = selectedIndex * segmentAngle;
        
        // Calculate rotation needed to center the selected segment
        const targetRotation = -selectedAngle - (segmentAngle / 2);
        const currentRotation = this.state.wheelRotation;
        
        // Animate rotation to center the selected segment
        d3.select(this.mainGroup.node())
          .transition()
          .duration(this.config.animationDuration)
          .attrTween('transform', () => {
            return t => {
              const interpolatedRotation = currentRotation + (targetRotation - currentRotation) * t;
              return `translate(${this.config.width / 2}, ${this.config.height / 2}) rotate(${interpolatedRotation * 180 / Math.PI})`;
            };
          })
          .on('end', () => {
            this.state.wheelRotation = targetRotation;
            
            // Animate segments out
            this.mainGroup.selectAll('.emotion-segment, .emotion-label')
              .transition()
              .duration(this.config.animationDuration / 2)
              .style('opacity', 0)
              .on('end', (d, i, nodes) => {
                // Only execute once when the first element's transition ends
                if (i === 0) {
                  // Update level and render new wheel
                  this.state.currentLevel += 1;
                  this.state.wheelRotation = 0;
                  this._renderWheel();
                  this.state.isTransitioning = false;
                }
              });
          });
      } else {
        // No animation, just update and render
        this.state.currentLevel += 1;
        this.state.wheelRotation = 0;
        this._renderWheel();
        this.state.isTransitioning = false;
      }
    } else {
      // This is a leaf emotion or we're at the tertiary level
      this.state.selectedEmotion = selectedEmotion;
      this.state.selectedIntensity = 0.5; // Default to middle intensity
      
      // Update UI to show selection
      d3.selectAll('.emotion-segment')
        .attr('stroke-width', function() {
          return this === element ? 3 : 1;
        })
        .attr('stroke', function() {
          return this === element ? self._getThemeColors().selectionStroke : self._getThemeColors().border;
        });
      
      // Add or update intensity indicator
      this._updateIntensityIndicator();
      
      // Call the selection callback
      this.config.onEmotionSelected(selectedEmotion, 0.5);
    }
  }

  /**
   * Handle center circle click (neutral emotion)
   * @private
   */
  _onCenterClick() {
    if (this.state.isTransitioning) return;
    
    // Define neutral emotion
    const neutralEmotion = {
      id: 'neutral',
      name: 'Neutral',
      valence: 0.5,
      arousal: 0.5
    };
    
    this.state.selectedEmotion = neutralEmotion;
    this.state.selectedIntensity = 0.5;
    
    // Reset segment styling
    d3.selectAll('.emotion-segment')
      .attr('stroke-width', 1)
      .attr('stroke', this._getThemeColors().border);
    
    // Highlight center
    d3.select('.emotion-wheel-center')
      .attr('stroke-width', 3)
      .attr('stroke', this._getThemeColors().selectionStroke);
    
    // Remove intensity indicator
    this.mainGroup.selectAll('.intensity-indicator').remove();
    
    // Call the selection callback
    this.config.onEmotionSelected(neutralEmotion, 0.5);
  }

  /**
   * Handle segment hover event
   * @param {Object} emotion - The hovered emotion
   * @param {Element} element - The hovered element
   * @param {Event} event - The mouse event
   * @private
   */
  _onSegmentHover(emotion, element, event) {
    // Highlight the hovered segment
    d3.select(element)
      .attr('stroke-width', 2)
      .attr('stroke', this._getThemeColors().hoverStroke);
    
    this.state.hoveredEmotion = emotion;
    
    // Show tooltip if enabled
    if (this.config.showTooltips && this.tooltip) {
      const [x, y] = d3.pointer(event, this.container);
      
      this.tooltip
        .style('visibility', 'visible')
        .style('opacity', 1)
        .style('left', `${x + 15}px`)
        .style('top', `${y}px`)
        .html(this._getTooltipContent(emotion));
    }
  }

  /**
   * Handle segment mouse leave event
   * @private
   */
  _onSegmentLeave() {
    // Reset segment styling unless it's the selected one
    d3.selectAll('.emotion-segment')
      .attr('stroke-width', d => {
        return this.state.selectedEmotion && d.id === this.state.selectedEmotion.id ? 3 : 1;
      })
      .attr('stroke', d => {
        return this.state.selectedEmotion && d.id === this.state.selectedEmotion.id ? 
          this._getThemeColors().selectionStroke : this._getThemeColors().border;
      });
    
    this.state.hoveredEmotion = null;
    
    // Hide tooltip
    if (this.config.showTooltips && this.tooltip) {
      this.tooltip
        .style('visibility', 'hidden')
        .style('opacity', 0);
    }
  }

  /**
   * Set up event handlers
   * @private
   */
  _setupEventHandlers() {
    // Resize handler
    if (this.config.responsive) {
      window.addEventListener('resize', this._handleResize.bind(this));
    }
    
    // Keyboard navigation
    this.container.addEventListener('keydown', this._handleKeyDown.bind(this));
  }

  /**
   * Handle resize event
   * @private
   */
  _handleResize() {
    // Debounce resize
    clearTimeout(this.resizeTimer);
    this.resizeTimer = setTimeout(() => {
      const containerRect = this.container.getBoundingClientRect();
      this.config.width = containerRect.width;
      this.config.height = containerRect.height;
      
      // Update SVG dimensions
      this.svg
        .attr('width', this.config.width)
        .attr('height', this.config.height);
      
      // Update main group position
      this.mainGroup
        .attr('transform', `translate(${this.config.width / 2}, ${this.config.height / 2})`);
      
      // Recalculate radius
      const minDimension = Math.min(this.config.width, this.config.height) - (this.config.margin * 2);
      this.radius = minDimension / 2;
      
      // Redraw wheel
      this._drawWheel();
      this._renderWheel();
    }, 250);
  }

  /**
   * Handle keyboard navigation
   * @param {KeyboardEvent} event - Keyboard event
   * @private
   */
  _handleKeyDown(event) {
    if (!this.state.selectedEmotion) return;
    
    switch (event.key) {
      case 'ArrowUp':
        // Increase intensity
        this.state.selectedIntensity = Math.min(1, this.state.selectedIntensity + 0.1);
        this._updateIntensityIndicator();
        this.config.onIntensityChange(this.state.selectedEmotion, this.state.selectedIntensity);
        event.preventDefault();
        break;
      case 'ArrowDown':
        // Decrease intensity
        this.state.selectedIntensity = Math.max(0, this.state.selectedIntensity - 0.1);
        this._updateIntensityIndicator();
        this.config.onIntensityChange(this.state.selectedEmotion, this.state.selectedIntensity);
        event.preventDefault();
        break;
      case 'ArrowLeft':
      case 'ArrowRight':
        // Move to the next/previous emotion
        const emotionsAtCurrentLevel = this._getEmotionsForCurrentLevel();
        const currentIndex = emotionsAtCurrentLevel.findIndex(e => 
          e.id === this.state.selectedEmotion.id);
        
        if (currentIndex !== -1) {
          const direction = event.key === 'ArrowLeft' ? -1 : 1;
          const newIndex = (currentIndex + direction + emotionsAtCurrentLevel.length) % 
            emotionsAtCurrentLevel.length;
          
          // Get the new emotion element and trigger click
          const newEmotionSelector = `.emotion-segment[data-emotion-id="${emotionsAtCurrentLevel[newIndex].id}"]`;
          const newEmotionElement = this.mainGroup.select(newEmotionSelector).node();
          
          if (newEmotionElement) {
            this._onSegmentClick(emotionsAtCurrentLevel[newIndex], newEmotionElement);
          }
        }
        event.preventDefault();
        break;
      case 'Escape':
        // Go back a level
        if (this.state.currentLevel > 0) {
          this._navigateBack();
        }
        event.preventDefault();
        break;
    }
  }

  /**
   * Make the wheel responsive
   * @private
   */
  _makeResponsive() {
    // Initial resize
    this._handleResize();
    
    // Add resize event listener if not already added
    if (!this._resizeListenerAdded) {
      window.addEventListener('resize', this._handleResize.bind(this));
      this._resizeListenerAdded = true;
    }
  }

  /**
   * Set up audio reactivity
   * @private
   */
  _setupAudioReactivity() {
    const audioAnalyzer = window.harmonicVAE.audioAnalyzer;
    
    // Animation frame loop
    const animate = () => {
      if (audioAnalyzer.isPlaying) {
        // Get audio data
        const audioData = audioAnalyzer.getFrequencyData();
        
        // Calculate average volume
        const avgVolume = audioData.reduce((sum, val) => sum + val, 0) / audioData.length;
        const normalizedVolume = avgVolume / 255;
        
        // Apply subtle pulse effect to the wheel based on volume
        const pulseScale = 1 + (normalizedVolume * 0.05);
        this.mainGroup.selectAll('.emotion-segment')
          .attr('transform', `scale(${pulseScale})`);
        
        // Apply subtle color intensity based on frequency bands
        const bassBand = audioData.slice(0, Math.floor(audioData.length * 0.1));
        const bassEnergy = bassBand.reduce((sum, val) => sum + val, 0) / bassBand.length / 255;
        
        this.mainGroup.selectAll('.emotion-segment')
          .attr('fill-opacity', 0.7 + (bassEnergy * 0.3));
      } else {
        // Reset transformations when audio is not playing
        this.mainGroup.selectAll('.emotion-segment')
          .attr('transform', '')
          .attr('fill-opacity', 1);
      }
      
      requestAnimationFrame(animate);
    };
    
    animate();
  }

  /**
   * Get tooltip content
   * @param {Object} emotion - The emotion to display
   * @private
   */
  _getTooltipContent(emotion) {
    let content = `<strong>${emotion.name}</strong>`;
    
    // Add valence and arousal if available
    if (emotion.valence !== undefined && emotion.arousal !== undefined) {
      content += `<br>Valence: ${(emotion.valence * 100).toFixed(0)}%`;
      content += `<br>Arousal: ${(emotion.arousal * 100).toFixed(0)}%`;
    }
    
    // Add description if available
    if (emotion.description) {
      content += `<br><br>${emotion.description}`;
    }
    
    // Add musical example if available
    if (emotion.musicalExample) {
      content += `<br><br>Example in music: ${emotion.musicalExample}`;
    }
    
    return content;
  }

  /**
   * Get the appropriate theme colors
   * @returns {Object} Theme colors object
   * @private
   */
  _getThemeColors() {
    const themes = {
      light: {
        background: '#f8f9fa',
        border: '#dee2e6',
        neutral: '#e9ecef',
        intensityLine: '#6c757d',
        handleFill: '#ffffff',
        handleStroke: '#495057',
        selectionStroke: '#212529',
        hoverStroke: '#343a40',
        buttonBackground: '#e9ecef',
        buttonText: '#212529'
      },
      dark: {
        background: '#212529',
        border: '#495057',
        neutral: '#343a40',
        intensityLine: '#adb5bd',
        handleFill: '#6c757d',
        handleStroke: '#ced4da',
        selectionStroke: '#f8f9fa',
        hoverStroke: '#e9ecef',
        buttonBackground: '#343a40',
        buttonText: '#f8f9fa'
      },
      colorful: {
        background: '#f1faee',
        border: '#457b9d',
        neutral: '#a8dadc',
        intensityLine: '#e63946',
        handleFill: '#ffffff',
        handleStroke: '#1d3557',
        selectionStroke: '#e63946',
        hoverStroke: '#1d3557',
        buttonBackground: '#a8dadc',
        buttonText: '#1d3557'
      }
    };
    
    return themes[this.config.theme] || themes.colorful;
  }

  /**
   * Get contrasting text color for background
   * @param {string} backgroundColor - Background color in hex
   * @returns {string} Contrasting text color (black or white)
   * @private
   */
  _getContrastColor(backgroundColor) {
    // Convert hex to RGB
    let hex = backgroundColor;
    if (hex.indexOf('#') === 0) {
      hex = hex.slice(1);
    }
    
    // Convert 3-digit hex to 6-digits
    if (hex.length === 3) {
      hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    }
    
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    
    // Calculate luminance using WCAG formula
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    
    return luminance > 0.5 ? '#000000' : '#ffffff';
  }

  /**
   * Get emotions for the current level
   * @returns {Array} List of emotions at the current level
   * @private
   */
  _getEmotionsForCurrentLevel() {
    if (this.state.currentLevel === 0) {
      return this.config.emotions;
    } else if (this.state.currentLevel === 1 && this.state.zoomedEmotion) {
      return this.state.zoomedEmotion.children || [];
    } else if (this.state.currentLevel === 2 && this.state.zoomedEmotion) {
      const parentEmotion = this.config.emotions.find(e => 
        e.children && e.children.some(child => child.id === this.state.zoomedEmotion.id));
      
      if (parentEmotion) {
        const secondaryEmotion = parentEmotion.children.find(e => e.id === this.state.zoomedEmotion.id);
        return secondaryEmotion.children || [];
      }
    }
    
    return [];
  }

  /**
   * Get the color for an emotion segment
   * @param {Object} emotion - The emotion object
   * @param {number} index - Index of the emotion
   * @param {number} total - Total number of emotions
   * @returns {string} Color in hex format
   * @private
   */
  _getEmotionColor(emotion, index, total) {
    // If emotion has explicit color, use it
    if (emotion.color) {
      return emotion.color;
    }
    
    // Otherwise calculate based on valence and arousal if available
    if (emotion.valence !== undefined && emotion.arousal !== undefined) {
      // Map valence (0-1) to hue (0-360)
      // Low valence (negative) = blue/purple (240), high valence (positive) = yellow/orange (60)
      const hue = 240 - (emotion.valence * 180);
      
      // Map arousal (0-1) to saturation and lightness
      // High arousal = high saturation, more vivid
      const saturation = 30 + (emotion.arousal * 70);
      
      // Medium arousal = medium lightness, high/low arousal = higher lightness (U-shaped)
      const arousalOffset = Math.abs(emotion.arousal - 0.5) * 30;
      const lightness = 45 + arousalOffset;
      
      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }
    
    // Default color scheme based on index
    return d3.interpolateRainbow(index / total);
  }

  /**
   * Get the font size for labels based on number of segments
   * @param {number} segmentCount - Number of segments
   * @returns {number} Font size in pixels
   * @private
   */
  _getLabelFontSize(segmentCount) {
    // Scale font size down as segments increase
    const baseFontSize = 14;
    return Math.max(8, baseFontSize - Math.log2(segmentCount));
  }

  /**
   * Default emotion structure based on the Geneva Emotion Wheel
   * @returns {Array} Default emotion structure
   * @private
   */
  _defaultEmotionStructure() {
    return [
      {
        id: 'joy',
        name: 'Joy',
        valence: 0.9,
        arousal: 0.7,
        color: '#FFD700', // Gold
        description: 'A feeling of great pleasure and happiness',
        musicalExample: 'Major keys, upbeat tempo',
        children: [
          {
            id: 'happiness',
            name: 'Happiness',
            valence: 0.85,
            arousal: 0.65,
            description: 'A state of wellbeing and contentment',
            children: [
              { id: 'contentment', name: 'Contentment', valence: 0.75, arousal: 0.4 },
              { id: 'satisfaction', name: 'Satisfaction', valence: 0.8, arousal: 0.5 },
              { id: 'serenity', name: 'Serenity', valence: 0.7, arousal: 0.3 }
            ]
          },
          {
            id: 'excitement',
            name: 'Excitement',
            valence: 0.9,
            arousal: 0.9,
            description: 'A feeling of great enthusiasm and eagerness',
            children: [
              { id: 'thrill', name: 'Thrill', valence: 0.95, arousal: 0.95 },
              { id: 'euphoria', name: 'Euphoria', valence: 1.0, arousal: 1.0 },
              { id: 'enthusiasm', name: 'Enthusiasm', valence: 0.85, arousal: 0.85 }
            ]
          }
        ]
      },
      {
        id: 'surprise',
        name: 'Surprise',
        valence: 0.6,
        arousal: 0.8,
        color: '#9370DB', // Medium Purple
        description: 'A feeling caused by something unexpected',
        musicalExample: 'Unexpected key changes, sudden dynamics',
        children: [
          {
            id: 'amazement',
            name: 'Amazement',
            valence: 0.7,
            arousal: 0.8,
            description: 'A feeling of great surprise or wonder',
            children: [
              { id: 'awe', name: 'Awe', valence: 0.75, arousal: 0.85 },
              { id: 'wonder', name: 'Wonder', valence: 0.8, arousal: 0.7 },
              { id: 'astonishment', name: 'Astonishment', valence: 0.65, arousal: 0.9 }
            ]
          },
          {
            id: 'confusion',
            name: 'Confusion',
            valence: 0.4,
            arousal: 0.7,
            description: 'A feeling of being bewildered or unclear',
            children: [
              { id: 'puzzlement', name: 'Puzzlement', valence: 0.45, arousal: 0.6 },
              { id: 'perplexity', name: 'Perplexity', valence: 0.35, arousal: 0.75 },
              { id: 'bewilderment', name: 'Bewilderment', valence: 0.3, arousal: 0.8 }
            ]
          }
        ]
      },
      {
        id: 'sadness',
        name: 'Sadness',
        valence: 0.1,
        arousal: 0.3,
        color: '#4682B4', // Steel Blue
        description: 'A feeling of sorrow or unhappiness',
        musicalExample: 'Minor keys, slow tempo, lower register',
        children: [
          {
            id: 'melancholy',
            name: 'Melancholy',
            valence: 0.25,
            arousal: 0.3,
            description: 'A feeling of pensive sadness',
            children: [
              { id: 'wistfulness', name: 'Wistfulness', valence: 0.3, arousal: 0.35 },
              { id: 'nostalgia', name: 'Nostalgia', valence: 0.4, arousal: 0.4 },
              { id: 'longing', name: 'Longing', valence: 0.3, arousal: 0.45 }
            ]
          },
          {
            id: 'despair',
            name: 'Despair',
            valence: 0.05,
            arousal: 0.2,
            description: 'A complete loss of hope',
            children: [
              { id: 'grief', name: 'Grief', valence: 0.1, arousal: 0.3 },
              { id: 'sorrow', name: 'Sorrow', valence: 0.15, arousal: 0.25 },
              { id: 'heartbreak', name: 'Heartbreak', valence: 0.05, arousal: 0.4 }
            ]
          }
        ]
      },
      {
        id: 'fear',
        name: 'Fear',
        valence: 0.2,
        arousal: 0.8,
        color: '#800000', // Maroon
        description: 'An unpleasant emotion caused by threat',
        musicalExample: 'Dissonance, tremolo strings, rhythmic uncertainty',
        children: [
          {
            id: 'anxiety',
            name: 'Anxiety',
            valence: 0.25,
            arousal: 0.75,
            description: 'A feeling of worry or nervousness',
            children: [
              { id: 'worry', name: 'Worry', valence: 0.3, arousal: 0.7 },
              { id: 'nervousness', name: 'Nervousness', valence: 0.35, arousal: 0.8 },
              { id: 'unease', name: 'Unease', valence: 0.3, arousal: 0.65 }
            ]
          },
          {
            id: 'terror',
            name: 'Terror',
            valence: 0.1,
            arousal: 0.95,
            description: 'Extreme fear',
            children: [
              { id: 'horror', name: 'Horror', valence: 0.15, arousal: 0.9 },
              { id: 'panic', name: 'Panic', valence: 0.2, arousal: 1.0 },
              { id: 'dread', name: 'Dread', valence: 0.05, arousal: 0.85 }
            ]
          }
        ]
      },
      {
        id: 'anger',
        name: 'Anger',
        valence: 0.15,
        arousal: 0.9,
        color: '#B22222', // Firebrick
        description: 'A strong feeling of annoyance or displeasure',
        musicalExample: 'Aggressive rhythms, loud dynamics, sharp articulations',
        children: [
          {
            id: 'irritation',
            name: 'Irritation',
            valence: 0.3,
            arousal: 0.7,
            description: 'Slight anger or annoyance',
            children: [
              { id: 'annoyance', name: 'Annoyance', valence: 0.35, arousal: 0.65 },
              { id: 'frustration', name: 'Frustration', valence: 0.25, arousal: 0.75 },
              { id: 'exasperation', name: 'Exasperation', valence: 0.3, arousal: 0.7 }
            ]
          },
          {
            id: 'rage',
            name: 'Rage',
            valence: 0.05,
            arousal: 1.0,
            description: 'Violent, uncontrollable anger',
            children: [
              { id: 'fury', name: 'Fury', valence: 0.1, arousal: 0.95 },
              { id: 'outrage', name: 'Outrage', valence: 0.15, arousal: 0.9 },
              { id: 'wrath', name: 'Wrath', valence: 0.05, arousal: 1.0 }
            ]
          }
        ]
      },
      {
        id: 'love',
        name: 'Love',
        valence: 0.9,
        arousal: 0.6,
        color: '#FF69B4', // Hot Pink
        description: 'A deep feeling of affection',
        musicalExample: 'Lyrical melodies, warm timbres, consonant harmonies',
        children: [
          {
            id: 'affection',
            name: 'Affection',
            valence: 0.85,
            arousal: 0.5,
            description: 'A gentle feeling of fondness',
            children: [
              { id: 'tenderness', name: 'Tenderness', valence: 0.8, arousal: 0.4 },
              { id: 'compassion', name: 'Compassion', valence: 0.75, arousal: 0.5 },
              { id: 'caring', name: 'Caring', valence: 0.8, arousal: 0.55 }
            ]
          },
          {
            id: 'passion',
            name: 'Passion',
            valence: 0.95,
            arousal: 0.85,
            description: 'Intense, driving feeling of love',
            children: [
              { id: 'desire', name: 'Desire', valence: 0.9, arousal: 0.9 },
              { id: 'adoration', name: 'Adoration', valence: 0.95, arousal: 0.7 },
              { id: 'infatuation', name: 'Infatuation', valence: 1.0, arousal: 0.8 }
            ]
          }
        ]
      },
      {
        id: 'peace',
        name: 'Peace',
        valence: 0.7,
        arousal: 0.2,
        color: '#00CED1', // Dark Turquoise
        description: 'A state of calm and tranquility',
        musicalExample: 'Ambient textures, gentle ostinatos, spacious arrangements',
        children: [
          {
            id: 'calm',
            name: 'Calm',
            valence: 0.65,
            arousal: 0.25,
            description: 'Free from disturbance or agitation',
            children: [
              { id: 'tranquility', name: 'Tranquility', valence: 0.7, arousal: 0.2 },
              { id: 'serenity', name: 'Serenity', valence: 0.75, arousal: 0.15 },
              { id: 'relaxation', name: 'Relaxation', valence: 0.65, arousal: 0.3 }
            ]
          },
          {
            id: 'harmony',
            name: 'Harmony',
            valence: 0.8,
            arousal: 0.3,
            description: 'A pleasing arrangement of parts',
            children: [
              { id: 'balance', name: 'Balance', valence: 0.75, arousal: 0.25 },
              { id: 'accord', name: 'Accord', valence: 0.8, arousal: 0.3 },
              { id: 'unity', name: 'Unity', valence: 0.85, arousal: 0.35 }
            ]
          }
        ]
      },
      {
        id: 'awe',
        name: 'Awe',
        valence: 0.7,
        arousal: 0.7,
        color: '#9932CC', // Dark Orchid
        description: 'A feeling of reverential respect mixed with fear or wonder',
        musicalExample: 'Grand orchestral textures, ethereal sounds, expansive range',
        children: [
          {
            id: 'reverence',
            name: 'Reverence',
            valence: 0.65,
            arousal: 0.6,
            description: 'Deep respect for someone or something',
            children: [
              { id: 'admiration', name: 'Admiration', valence: 0.7, arousal: 0.55 },
              { id: 'veneration', name: 'Veneration', valence: 0.6, arousal: 0.65 },
              { id: 'respect', name: 'Respect', valence: 0.65, arousal: 0.5 }
            ]
          },
          {
            id: 'transcendence',
            name: 'Transcendence',
            valence: 0.8,
            arousal: 0.75,
            description: 'Beyond ordinary experience',
            children: [
              { id: 'spirituality', name: 'Spirituality', valence: 0.75, arousal: 0.7 },
              { id: 'elevation', name: 'Elevation', valence: 0.85, arousal: 0.8 },
              { id: 'sublimity', name: 'Sublimity', valence: 0.9, arousal: 0.85 }
            ]
          }
        ]
      }
    ];
  }

  /**
   * Public API methods
   */

  /**
   * Get the currently selected emotion and intensity
   * @returns {Object} Object containing emotion and intensity
   */
  getSelection() {
    return {
      emotion: this.state.selectedEmotion,
      intensity: this.state.selectedIntensity
    };
  }

  /**
   * Set the selected emotion programmatically
   * @param {string} emotionId - ID of the emotion to select
   * @param {number} intensity - Intensity value (0-1)
   * @returns {boolean} Success flag
   */
  setSelection(emotionId, intensity = 0.5) {
    // Find the emotion in our structure
    let emotion = null;
    let level = 0;
    let foundInLevel = -1;
    
    // Search in primary emotions
    emotion = this.config.emotions.find(e => e.id === emotionId);
    if (emotion) foundInLevel = 0;
    
    // Search in secondary emotions
    if (!emotion) {
      for (const primary of this.config.emotions) {
        if (primary.children) {
          emotion = primary.children.find(e => e.id === emotionId);
          if (emotion) {
            foundInLevel = 1;
            this.state.zoomedEmotion = primary;
            break;
          }
        }
      }
    }
    
    // Search in tertiary emotions
    if (!emotion) {
      for (const primary of this.config.emotions) {
        if (primary.children) {
          for (const secondary of primary.children) {
            if (secondary.children) {
              emotion = secondary.children.find(e => e.id === emotionId);
              if (emotion) {
                foundInLevel = 2;
                this.state.zoomedEmotion = secondary;
                break;
              }
            }
          }
          if (emotion) break;
        }
      }
    }
    
    if (!emotion) return false;
    
    // Set the state
    this.state.selectedEmotion = emotion;
    this.state.selectedIntensity = Math.min(1, Math.max(0, intensity));
    this.state.currentLevel = foundInLevel;
    
    // Update the wheel
    this._renderWheel();
    
    // Call the selection callback
    this.config.onEmotionSelected(emotion, intensity);
    
    return true;
  }

  /**
   * Reset the emotion wheel
   */
  reset() {
    this.state = {
      selectedEmotion: null,
      selectedIntensity: 0,
      currentLevel: 0,
      hoveredEmotion: null,
      wheelRotation: 0,
      zoomedEmotion: null,
      isTransitioning: false
    };
    
    this._renderWheel();
  }

  /**
   * Update the configuration
   * @param {Object} newConfig - New configuration options
   */
  updateConfig(newConfig) {
    this.config = {
      ...this.config,
      ...newConfig
    };
    
    this._renderWheel();
  }

  /**
   * Destroy the wheel and clean up
   */
  destroy() {
    // Remove event listeners
    window.removeEventListener('resize', this._handleResize.bind(this));
    this.container.removeEventListener('keydown', this._handleKeyDown.bind(this));
    
    // Remove SVG and tooltip
    this.svg.remove();
    if (this.tooltip) this.tooltip.remove();
    
    // Clear container
    this.container.innerHTML = '';
  }
}
