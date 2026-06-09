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
import { CuisineChoices } from "../constants/cuisine";

export function SearchBar() {
  const [dishName, setDishName] = useState("");
  const [cuisine, setCuisine] = useState("");
  const { results, loading, error, search } = useSearch();

  function handleSubmit(e) {
    e.preventDefault();
    search(dishName, cuisine);
  }

  return (
    <div>
      <form onSubmit={handleSubmit} className="d-flex gap-2 flex-wrap mb-4">
        <input
          type="text"
          className="form-control"
          placeholder="Dish name, e.g. lasagna"
          value={dishName}
          onChange={(e) => setDishName(e.target.value)}
          required
          aria-label="Dish name"
        />
        <select
          className="form-select"
          value={cuisine}
          onChange={(e) => setCuisine(e.target.value)}
          aria-label="Cuisine type"
          style={{ maxWidth: 200 }}
        >
          <option value="">Any cuisine</option>
          {CuisineChoices.map((c) => (
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
          {loading ? "Searching…" : "Find recipes"}
        </button>
      </form>

      {error && (
        <div className="alert alert-danger" role="alert">
          {error}
        </div>
      )}

      {results.length > 0 && <RecipeResults results={results} />}
    </div>
  );
}
