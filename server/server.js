/**
 * Backend server for HarmonicVAE feedback system
 * Uses Express and SQLite for database management
 */
const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const winston = require('winston'); // For logging
const rateLimit = require('express-rate-limit'); // For rate limiting
const helmet = require('helmet'); // For security headers
const { body, validationResult } = require('express-validator'); // For input validation

// Create Express app
const app = express();
const port = process.env.PORT || 3000;

// Configure logging
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
  ],
});

if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.simple(),
  }));
}

// Middleware
app.use(helmet()); // Security headers
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// Rate limiting to prevent abuse
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);

// Connect to SQLite database
const db = new sqlite3.Database('./harmonic_vae_feedback.db', (err) => {
  if (err) {
    logger.error('Error connecting to database:', err);
  } else {
    logger.info('Connected to SQLite database');
    createTables();
  }
});

// Create database tables if they don’t exist
function createTables() {
  db.run(`
    CREATE TABLE IF NOT EXISTS feedback (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      generation_id TEXT NOT NULL,
      emotion_name TEXT NOT NULL,
      emotion_valence REAL NOT NULL,
      emotion_arousal REAL NOT NULL,
      quality_rating INTEGER NOT NULL,
      comments TEXT,
      listen_duration REAL,
      consent BOOLEAN NOT NULL,
      timestamp TEXT NOT NULL
    )
  `, (err) => {
    if (err) {
      logger.error('Error creating feedback table:', err);
    } else {
      logger.info('Feedback table ready');
    }
  });

  db.run(`
    CREATE TABLE IF NOT EXISTS generation_data (
      generation_id TEXT PRIMARY KEY,
      latent_vector BLOB,
      condition_params TEXT,
      timestamp TEXT NOT NULL
    )
  `, (err) => {
    if (err) {
      logger.error('Error creating generation_data table:', err);
    } else {
      logger.info('Generation data table ready');
    }
  });
}

// Helper functions for database queries with promises
function dbRun(query, params) {
  return new Promise((resolve, reject) => {
    db.run(query, params, function (err) {
      if (err) reject(err);
      else resolve(this);
    });
  });
}

function dbAll(query, params) {
  return new Promise((resolve, reject) => {
    db.all(query, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows);
    });
  });
}

function dbGet(query, params) {
  return new Promise((resolve, reject) => {
    db.get(query, params, (err, row) => {
      if (err) reject(err);
      else resolve(row);
    });
  });
}

/**
 * API Routes
 */

// Submit feedback with input validation
app.post('/api/feedback', [
  body('generation_id').isString().notEmpty(),
  body('emotion.name').isString().notEmpty(),
  body('emotion.valence').isFloat({ min: 0, max: 1 }),
  body('emotion.arousal').isFloat({ min: 0, max: 1 }),
  body('quality_rating').isInt({ min: 1, max: 5 }),
  body('comments').optional().isString(),
  body('listen_duration').optional().isFloat({ min: 0 }),
  body('consent').isBoolean(),
], async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ success: false, errors: errors.array() });
  }

  const data = req.body;
  const timestamp = new Date().toISOString();

  try {
    await dbRun('BEGIN TRANSACTION');

    // Insert or replace generation data
    await dbRun(`
      INSERT OR REPLACE INTO generation_data (
        generation_id, latent_vector, condition_params, timestamp
      ) VALUES (?, ?, ?, ?)
    `, [
      data.generation_id,
      JSON.stringify(data.latent_vector),
      JSON.stringify(data.condition_params),
      timestamp,
    ]);

    // Insert feedback
    await dbRun(`
      INSERT INTO feedback (
        generation_id, emotion_name, emotion_valence, emotion_arousal,
        quality_rating, comments, listen_duration, consent, timestamp
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `, [
      data.generation_id,
      data.emotion.name,
      data.emotion.valence,
      data.emotion.arousal,
      data.quality_rating,
      data.comments || '',
      data.listen_duration || 0,
      data.consent,
      timestamp,
    ]);

    await dbRun('COMMIT');
    logger.info(`Feedback submitted for generation_id: ${data.generation_id}`);
    res.status(200).json({ success: true, message: 'Feedback saved successfully' });
  } catch (error) {
    await dbRun('ROLLBACK');
    logger.error('Error saving feedback:', error);
    res.status(500).json({ success: false, message: 'Error saving feedback' });
  }
});

// Get feedback statistics
app.get('/api/stats', async (req, res) => {
  try {
    const [emotionStats, ratingStats, avgRating, totalCount] = await Promise.all([
      dbAll(`
        SELECT emotion_name, COUNT(*) as count
        FROM feedback
        GROUP BY emotion_name
        ORDER BY count DESC
      `),
      dbAll(`
        SELECT quality_rating, COUNT(*) as count
        FROM feedback
        GROUP BY quality_rating
        ORDER BY quality_rating ASC
      `),
      dbGet(`SELECT AVG(quality_rating) as average_rating FROM feedback`),
      dbGet(`SELECT COUNT(*) as total FROM feedback`),
    ]);

    res.status(200).json({
      success: true,
      emotions: emotionStats,
      ratings: ratingStats,
      average_rating: avgRating.average_rating,
      total_feedback: totalCount.total,
    });
  } catch (error) {
    logger.error('Error getting statistics:', error);
    res.status(500).json({ success: false, message: 'Error getting statistics' });
  }
});

// Start server
app.listen(port, () => {
  logger.info(`HarmonicVAE feedback server running on port ${port}`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  db.close((err) => {
    if (err) {
      logger.error('Error closing database:', err);
    } else {
      logger.info('Database connection closed');
    }
    process.exit(0);
  });
});
