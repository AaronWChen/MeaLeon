/**
 * hooks/useSearch.js
 *
 * Custom hook that owns all search state — loading, error, results.
 * The component just calls search(dish, cuisine) and reads back state.
 *
 * Why a custom hook instead of putting fetch() in the component:
 *   - State logic is reusable (search bar on index + results page both use it)
 *   - Easy to test in isolation
 *   - Component stays declarative — it describes UI, not data fetching
 */

import { useState, useCallback } from "react";
import { getRecommendations } from "../api/client";

export function useSearch() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Stable reference — won't cause re-renders if passed as a prop
  const search = useCallback(async (dishName, cuisine = "") => {
    if (!dishName.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const data = await getRecommendations({ dishName, cuisine });
      setResults(data.results ?? []);
    } catch (err) {
      setError(err.message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
    setError(null);
  }, []);

  return { results, loading, error, search, clearResults };
}
