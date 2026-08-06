const BASE_URL = 'http://localhost:8000';

/**
 * Sends a request to start generating a puzzle.
 * 
 * @param {Object} params
 * @param {number} params.rating - The target ELO rating (e.g., 1500)
 * @param {string[]} params.themes - Array of selected chess themes
 * @returns {Promise<{ jobId: string }>} The created job details
 */
export async function generatePuzzle({ rating, themes }) {
  const response = await fetch(`${BASE_URL}/api/puzzles/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      rating,
      themes,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to generate puzzle: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Checks the status of an ongoing puzzle generation job.
 * 
 * @param {string} jobId - The UUID of the job
 * @returns {Promise<{ status: 'pending' | 'completed' | 'failed' | 'cancelled', puzzle?: any }>}
 */
export async function checkPuzzleStatus(jobId) {
  const response = await fetch(`${BASE_URL}/api/puzzles/status/${jobId}`);
  if (!response.ok) {
    throw new Error(`Failed to check status for job ${jobId}: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Tells the backend to cancel puzzle generation for an active job.
 * Uses keepalive to ensure the request completes even if the page is being closed.
 * 
 * @param {string} jobId - The UUID of the job
 * @returns {Promise<{ status: string }>}
 */
export async function cancelPuzzleGeneration(jobId) {
  try {
    const response = await fetch(`${BASE_URL}/api/puzzles/cancel/${jobId}`, {
      method: 'POST',
      keepalive: true, // Crucial for reliable trigger during tab/page close
      headers: {
        'Content-Type': 'application/json',
      },
    });
    if (!response.ok) {
      console.warn(`Cancel request failed with status: ${response.status}`);
    }
    return response.json();
  } catch (error) {
    console.error(`Failed to send cancel request for job ${jobId}:`, error);
    return { status: 'error', message: error.message };
  }
}
