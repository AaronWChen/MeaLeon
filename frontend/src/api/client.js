/**
 * api/client.js
 *
 * Central API client for all Flask backend calls.
 * The frontend never calls search_service or recommend_service directly —
 * everything goes through /api/* on the Flask backend (Option B).
 *
 * Why a single client module:
 *   - One place to set the auth header (Bearer token from localStorage)
 *   - One place to handle 401s (redirect to login)
 *   - One place to change the base URL (env var swap between dev/prod)
 *   - Components stay clean — they call functions, not fetch()
 */

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

// ---------------------------------------------------------------------------
// Core fetch wrapper
// ---------------------------------------------------------------------------

async function apiFetch(path, options = {}) {
  const token = localStorage.getItem("token");

  const response = await fetch(`${BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
    ...options,
  });

  // Token expired or invalid — clear it and redirect to login
  if (response.status === 401) {
    localStorage.removeItem("token");
    window.location.href = "/login";
    return;
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: response.statusText }));
    throw new Error(error.message || `API error ${response.status}`);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export async function login(username, password) {
  const data = await apiFetch("/api/tokens", {
    method: "POST",
    // Basic auth for token endpoint (matches existing tokens.py)
    headers: {
      Authorization: "Basic " + btoa(`${username}:${password}`),
    },
    body: JSON.stringify({}),
  });
  localStorage.setItem("token", data.token);
  return data;
}

export function logout() {
  localStorage.removeItem("token");
}

export function isLoggedIn() {
  return Boolean(localStorage.getItem("token"));
}

// ---------------------------------------------------------------------------
// Recommendations — the main feature
// ---------------------------------------------------------------------------

/**
 * Ask the backend for recipe recommendations.
 *
 * The backend will:
 *   1. Verify the JWT and load the user
 *   2. Fetch their preferences (diet, allergies) from Postgres
 *   3. Call search_service with dish + cuisine
 *   4. Call recommend_service with ingredients + user context
 *   5. Filter results against user restrictions
 *   6. Return the merged, filtered list
 *
 * Unauthenticated users get recommendations too — they just don't get
 * personalised filtering. The backend handles this gracefully.
 */
export async function getRecommendations({ dishName, cuisine = "" }) {
  return apiFetch("/api/recommend", {
    method: "POST",
    body: JSON.stringify({ dish_name: dishName, cuisine }),
  });
}

// ---------------------------------------------------------------------------
// User preferences
// ---------------------------------------------------------------------------

export async function getUserPreferences() {
  return apiFetch("/api/users/me/preferences");
}

export async function updateUserPreferences(preferences) {
  return apiFetch("/api/users/me/preferences", {
    method: "PUT",
    body: JSON.stringify(preferences),
  });
}

// ---------------------------------------------------------------------------
// Reviews (existing functionality — no change to routes)
// ---------------------------------------------------------------------------

export async function getUserReviews(username, page = 1) {
  return apiFetch(`/api/users/${username}/reviews?page=${page}`);
}

export async function submitReview(review) {
  return apiFetch("/api/reviews", {
    method: "POST",
    body: JSON.stringify(review),
  });
}
