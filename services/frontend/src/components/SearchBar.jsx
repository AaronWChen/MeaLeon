/**
 * components/SearchBar.jsx
 *
 * Search form — dish name + optional cuisine selector.
 * Calls the useSearch hook on submit; renders results below.
 *
 * Kept intentionally simple. Styling uses your existing bootstrap.min.css.
 */

import { useState } from "react";
import { useSearch } from "../hooks/useSearch";
import { RecipeResults } from "./RecipeResults";

const CUISINE_CHOICES = [
  { value: "american", label: "American" },
  { value: "asian", label: "Asian" },
  { value: "british", label: "British" },
  { value: "caribbean", label: "Caribbean" },
  { value: "central europe", label: "Central European" },
  { value: "chinese", label: "Chinese" },
  { value: "eastern europe", label: "Eastern European" },
  { value: "french", label: "French" },
  { value: "greek", label: "Greek" },
  { value: "indian", label: "Indian" },
  { value: "italian", label: "Italian" },
  { value: "japanese", label: "Japanese" },
  { value: "korean", label: "Korean" },
  { value: "kosher", label: "Kosher" },
  { value: "mediterranean", label: "Mediterranean" },
  { value: "mexican", label: "Mexican" },
  { value: "middle eastern", label: "Middle Eastern" },
  { value: "nordic", label: "Nordic" },
  { value: "south american", label: "South American" },
  { value: "south east asian", label: "South East Asian" },
  { value: "world", label: "World" },
];

export function SearchBar() {
  const [dishName, setDishName] = useState("");
  const [cuisine, setCuisine] = useState("");
  const [hasSearched, setHasSearched] = useState(false);
  const {
    results,
    loading,
    error,
    source,
    topQueryIngredients,
    query,
    search,
  } = useSearch();

  function handleSubmit(e) {
    e.preventDefault();
    setHasSearched(true);
    search(dishName, cuisine);
  }


  return (
    <div>
      <form onSubmit={handleSubmit} className="d-flex gap-2 flex-wrap mb-4">
        <input
          type="text"
          className="form-control"
          id="dish-name"
          placeholder="Dish name, e.g. lasagna"
          value={dishName}
          onChange={(e) => setDishName(e.target.value)}
          required
          aria-label="Dish name"
          style={{ minWidth: 200, flex: 1 }}
        />
        <select
          className="form-select"
          value={cuisine}
          id="cuisine"
          onChange={(e) => setCuisine(e.target.value)}
          aria-label="Cuisine type"
          style={{ maxWidth: 200 }}
        >
          <option value="">Any cuisine</option>
          {CUISINE_CHOICES.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
        </select>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={loading}
        >
          {loading ? "Searching…" : "Find similar recipes"}
        </button>
      </form>

      {error && (
        <div className="alert alert-danger" role="alert">
          {error}
        </div>
      )}

      {results.length > 0 && (
        <>
          {source === "edamam_fallback" && (
            <div className="alert alert-info py-2 small mb-3">
              Showing results from Edamam — local recipe index still loading.
            </div>
          )}
          <RecipeResults
            results={results}
            query={query}
            topQueryIngredients={topQueryIngredients}
          />
        </>
      )}

      {!loading && !error && results.length === 0 && hasSearched && (
        <p className="text-muted">No results found. Try a different dish or cuisine.</p>
      )}
    </div>
  );
}
