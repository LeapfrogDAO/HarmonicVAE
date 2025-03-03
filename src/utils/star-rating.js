/**
 * Creates an interactive 5-star rating component using D3.js
 * @param {string} containerId - The ID of the container element where the rating system will be rendered
 * @returns {Object} - Public API with methods to get, set, and reset the rating
 */
function createStarRating(containerId) {
  // **Define Dimensions and Constants**
  const width = 250;         // Total width of the SVG container
  const height = 50;         // Height of the SVG container
  const starSpacing = 40;    // Spacing between star centers
  const starSize = 300;      // Area of each star symbol (adjustable)

  // **Create SVG Container**
  const svg = d3.select(`#${containerId}`)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("class", "star-rating");

  // **Define Star Symbol**
  const starSymbol = d3.symbol()
    .type(d3.symbolStar)
    .size(starSize);

  // **Define Glow Effect Filter**
  const defs = svg.append("defs");
  const glowFilter = defs.append("filter")
    .attr("id", "star-glow")
    .attr("x", "-50%")      // Extend filter bounds to accommodate blur
    .attr("y", "-50%")
    .attr("width", "200%")
    .attr("height", "200%");

  glowFilter.append("feGaussianBlur")
    .attr("stdDeviation", "3")  // Blur intensity for glow
    .attr("result", "coloredBlur");

  const feMerge = glowFilter.append("feMerge");
  feMerge.append("feMergeNode").attr("in", "coloredBlur");
  feMerge.append("feMergeNode").attr("in", "SourceGraphic"); // Combine blur with original

  // **State Variables**
  let currentRating = -1;  // -1 means no rating selected (0-4 for stars)
  let hoverRating = -1;    // Tracks hovered star index

  // **Create Stars**
  const stars = svg.selectAll(".star")
    .data(d3.range(5))     // Array [0, 1, 2, 3, 4] for 5 stars
    .enter()
    .append("path")
    .attr("class", "star")
    .attr("d", starSymbol)
    .attr("transform", (d, i) => `translate(${starSpacing * i + starSpacing / 2}, ${height / 2})`)
    .attr("fill", "#ccc")          // Default unselected color (gray)
    .attr("stroke", "#888")        // Subtle border
    .attr("stroke-width", 1)
    .attr("cursor", "pointer")     // Indicate interactivity
    .on("mouseover", handleMouseOver)
    .on("mouseout", handleMouseOut)
    .on("click", handleClick);

  // **Event Handlers**
  function handleMouseOver(event, i) {
    hoverRating = i;
    updateStars();
  }

  function handleMouseOut() {
    hoverRating = -1;
    updateStars();
  }

  function handleClick(event, i) {
    currentRating = i;
    updateStars();
    // Apply pulse animation to clicked star
    d3.select(event.currentTarget)
      .transition()
      .duration(150)
      .attr("d", d3.symbol().type(d3.symbolStar).size(starSize * 1.3)())
      .transition()
      .duration(150)
      .attr("d", starSymbol);
  }

  // **Update Stars Appearance**
  function updateStars() {
    const activeRating = hoverRating >= 0 ? hoverRating : currentRating;
    stars.attr("fill", (d, i) => i <= activeRating ? "#FFD700" : "#ccc")  // Gold for active, gray for inactive
         .attr("filter", (d, i) => i <= activeRating ? "url(#star-glow)" : null); // Glow on active stars
    updateRatingDisplay(activeRating >= 0 ? activeRating + 1 : null);
  }

  // **Update Rating Display**
  function updateRatingDisplay(rating) {
    const displayElement = document.getElementById("rating-value");
    if (displayElement) {
      displayElement.textContent = rating === null ? "No rating yet" : `${rating} out of 5 stars`;
    }
  }

  // **Initialize Display**
  updateRatingDisplay(null);

  // **Public API**
  return {
    /**
     * Gets the current rating (1-5) or null if no rating is selected
     * @returns {number|null}
     */
    getRating: () => currentRating >= 0 ? currentRating + 1 : null,

    /**
     * Programmatically sets the rating
     * @param {number} rating - Rating value between 1 and 5
     */
    setRating: (rating) => {
      if (rating >= 1 && rating <= 5) {
        currentRating = rating - 1;
        hoverRating = -1;  // Reset hover state
        updateStars();
      }
    },

    /**
     * Resets the rating to unselected state
     */
    resetRating: () => {
      currentRating = -1;
      hoverRating = -1;
      updateStars();
    }
  };
}
